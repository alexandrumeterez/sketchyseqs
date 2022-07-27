from __future__ import annotations

import tqdm
import time
import editdistance 
import matplotlib.pyplot as plt
import random

import numpy as np
import numba as nb
from numba import njit, jit, prange, cuda, objmode
from numba.typed import List
from numba.experimental import jitclass


@njit(fastmath=True)
def l2_dist(a, b): 
    d = a-b
    return np.sum(d*d);


@njit(fastmath=True)
def hamming_distance(a, b):
    return np.sum(a != b);


# only works for k<=32, would overflow otherwise 
@njit(fastmath=True)
def extract_kmers(seq,k):
    kmer = 0
    kmers = np.zeros(len(seq)-k+1,dtype=np.int64)
    for i,c in enumerate(seq):
        kmer = kmer * 4 + c
        kmer = kmer % (4**k)
        if i>=k:
            kmers[i-k] = kmer
    return kmers


# Simple implementation of MinHash sketching for kmers
# only works for k<=32, would overflow otherwise 
@jitclass([
    ('A', nb.int32),
    ('k', nb.int32),
    ('D', nb.int32),
    ('hashes', nb.int64[:, :])])
class MH():
    def __init__(self, D, k, A):
        self.A = A
        self.D = D
        self.k = k
        len = A**k
        self.hashes = np.empty((self.D, len), dtype=np.int64)
        for d in range(self.D):
            for i in range(len):
                self.hashes[d][i] = random.randint(0,len) # 10x to avoid collisions

    def sketch_one(self, seq):
        kmers = extract_kmers(seq, self.k)
        sketch = np.empty((self.D),dtype=np.int64)
        for d in range(self.D):
            MH = -1
            for k in kmers:
                if MH<0 or self.hashes[d][k]<MH:
                    MH = self.hashes[d][k]
            sketch[d] = MH
        return sketch
    
    def sketch(self, seqs):
        return [self.sketch_one(s) for s in seqs]
    
    def dist(self, s1, s2):
        return hamming_distance(s1, s2)
        

@jitclass([
    ('A', nb.int32),
    ('t', nb.int32),
    ('D', nb.int32),
    ('normalize', nb.bool_),
    ('hashes', nb.int32[:, :]),
    ('signs', nb.float32[:, :])])
class TS():
    def __init__(self, t, D, A, normalize = True):
        self.A = A
        self.t = t
        self.D = D
        self.normalize = normalize

        # An A*t array of random integers in [0, D)
        self.hashes = np.empty((self.A, self.t), dtype=np.int32)
        # An A*t array of random +-1
        self.signs = np.empty((self.A, self.t), dtype=np.float32)
        for c in range(self.A):
            for k in range(self.t):
                self.hashes[c][k] = random.randrange(0, self.D)
                self.signs[c][k] = random.randrange(-1, 2, 2)

    def _full_sketch(self, seq: nb.int32[:]):
        # NOTE: The sketch is stored as float64 here so counting won't overflow.
        T = np.zeros((self.t + 1, self.D), dtype=np.float64)
        T[0][0] = 1
        
        for c in seq:
            for k in range(self.t - 1, -1, -1):
                h = self.hashes[c][k]
                s = self.signs[c][k]
                for l in range(self.D):
                    r = l + h if l + h < self.D else l + h - self.D
                    T[k + 1][l] += s * T[k][r]

        return T

    def _normalize(self, seq, T):
        if self.normalize:
            # # Normalization factor.
            # n = len(seq)
            # nct = nb.float64(1)
            # for i in range(self.t):
            #     nct = nct * (n - i) / (i + 1)
            # T /= nct
            T = T / np.linalg.norm(T)
        return T

    def sketch_one(self, seq: np.ndarray) -> nb.float32[:]:
        full_sketch = self._full_sketch(seq)

        sketch = self._normalize(seq, full_sketch[self.t])

        sketch = sketch.flatten()
        return sketch

    def sketch(self, seqs):
        return [self.sketch_one(seq) for seq in seqs]

    def dist(self, s1, s2):
        return l2_dist(s1,s2)


# a_1...a_t is mapped to index  A^{t-1} a_1 + ... + A * a_{t-1} + 1 * a_t
@jitclass([
           ('pow', nb.int32[:]),
           ('A', nb.int32),
           ('t', nb.int32),
           ('normalize', nb.bool_),])
class TE():
    # https://github.com/numba/numba/issues/1694

    def __init__(self, t, A = 4, normalize = True):
        self.t = t
        self.A = A
        self.normalize = normalize

        self.pow = np.zeros(self.t + 1, np.int32)
        self.pow[0] = 1
        for i in range(1, self.t + 1):
            self.pow[i] = self.A * self.pow[i - 1]

    # NOTE: The sketch is stored as float64 here so counting won't overflow.
    def _empty_tensor(self):
        Ts = List()
        for l in self.pow:
            Ts.append(np.zeros(l, np.float64))
        return Ts

    # Return the sketch for the concatenation of two sequences.
    # TODO: Optimize this to modify Tr in place.
    def _join(self, Tl, Tr):
        Ts = self._empty_tensor()
        for tr in range(self.t + 1):
            for tl in range(self.t + 1 - tr):
                Ts[tl + tr] += np.kron(Tl[tl], Tr[tr])
        return Ts

    # Returns the raw 1D count sketches for all tuple sizes up to t.
    # NOTE: This returns counts, not frequencies.
    def _full_sketch(self, seq: nb.int32[:]):
        Ts = self._empty_tensor()

        Ts[0][0] = 1

        # sketch
        for c in seq:
            assert 0 <= c and c < self.A
            for i in range(self.t - 1, -1, -1):
                for j in range(len(Ts[i])):
                    Ts[i + 1][self.A * j + c] += Ts[i][j]
        return Ts

    def sketch_one(self, seq: nb.int32[:]) -> nb.float32[:]:
        full_sketch = self._full_sketch(seq)
        if self.normalize:
            # Normalization factor.
            n = len(seq)
            nct = nb.float64(1)
            for i in range(self.t):
                nct = nct * (n - i) / (i + 1)
            full_sketch[self.t] /= nct
        sketch = np.array([x for x in full_sketch[self.t]], dtype=nb.float32)
        return sketch

    # Returns the sketch for the given t as frequencies.
    def sketch(self, seqs: list[nb.int32[:]]) -> list[nb.float32[:]]:
        return [self.sketch_one(seq) for seq in seqs]

    def dist(self, s1, s2):
        return l2_dist(s1, s2)


# code for GPU tensor sketch 
@cuda.jit(fastmath=True)
def _gpu_sketch(A, t, D, L, hashes, signs, seq, starts, T):
    seqid = cuda.blockIdx.x
    start = starts[seqid]
    end = starts[seqid + 1]

    l = cuda.threadIdx.x
    k = cuda.threadIdx.y
    assert k < t
    assert l < D // L

    # We use a 2*(t+1)*D tensor consisting of two 'planes'.
    # At each step, one plane is the input, and one is the output. Which is indicated by `j` further down.
    plane = (t + 1) * D
    threads = t * D // L

    # Slice the shared memory into local shared memory arrays.
    # Note the different types per view.

    # NOTE: Tin has a variable offset of k*D to save a bit on further computations.
    Tin = cuda.shared.array(shape=0, dtype=nb.float32)[k * D : 2 * plane]
    local_seq = cuda.shared.array(shape=0, dtype=nb.int32)[2 * plane : 2 * plane + threads]

    local_signs = cuda.shared.array(shape=0, dtype=nb.float32)[
        2 * plane + threads : 2 * plane + threads + A * t
    ]
    local_hashes = cuda.shared.array(shape=0, dtype=nb.int32)[
        2 * plane + threads + A * t : 2 * plane + threads + 2 * A * t
    ]

    # Copy the device memory hashes/signs to shared memory.
    if l < A:
        local_hashes[l * t + k] = hashes[l][k]
        local_signs[l * t + k] = signs[l][k]

    # Initialize the tensors to 0.
    for ll in range(l, D, D // L):
        Tin[0 * plane + 0 * D + ll] = 0
        Tin[0 * plane + (0 + 1) * D + ll] = 0
        Tin[1 * plane + 0 * D + ll] = 0
        Tin[1 * plane + (0 + 1) * D + ll] = 0

    cuda.syncthreads()

    # Initialize the 0-element of the tensor to 1.
    if k == 0:
        Tin[0] = 1
        Tin[plane] = 1

    cuda.syncthreads()

    # The offset for the plane we're currently reading from. The write offset
    # is the other plane: `plane-read_plane`.
    read_plane = 0

    # Loop over characters in the sequence.
    tid = l + k * D // L
    for i in range((end - start) // threads):
        # Read `threads` characters from `seq` and store them in `local_seq` in shared memory.
        idx = start + i * threads + tid
        local_seq[tid] = seq[idx]
        cuda.syncthreads()

        # Process the fetched characters.
        for c in local_seq:
            h = local_hashes[c * t + k]
            s = local_signs[c * t + k]
            write_plane = plane - read_plane
            # Process L consecutive indices (of the D in total).
            # 0 <= l < D/L, so this covers all of [0, D).
            for ll in range(L * l, L * (l + 1)):
                # Compute the shifted target index, avoiding a modulo operation.
                r = ll + h
                r -= D if r >= D else 0
                # Write to output tensor.
                Tin[write_plane + D + ll] = Tin[read_plane + D + ll] + s * Tin[read_plane + r]

            # After this thread has processed the current character `c`, swap the active plane and wait for other threads.
            read_plane = write_plane
            cuda.syncthreads()

    # Process the remaining characters. We don't do synchronous prefetching to
    # shared memory here, because this only covers the last few characters of
    # the sequence.
    # TODO: If sequences are short, it may actually be beneficial to still do this.
    for idx in range(start + (end - start) // threads * threads, end):
        c = seq[idx]
        # Same code as above.
        h = local_hashes[c * t + k]
        s = local_signs[c * t + k]
        write_plane = plane - read_plane
        for ll in range(L * l, L * (l + 1)):
            r = ll + h
            r -= D if r >= D else 0
            Tin[write_plane + D + ll] = Tin[read_plane + D + ll] + s * Tin[read_plane + r]

        read_plane = write_plane
        cuda.syncthreads()

    # Copy to result.
    for ll in range(l, D, D // L):
        T[seqid][k][ll] = Tin[read_plane + ll]
        T[seqid][k + 1][ll] = Tin[read_plane + D + ll]


class cuTS:
    def __init__(self, A, t, D, L=1, normalize=True):
        self.A = A
        self.t = t
        self.D = D
        self.normalize = normalize
        
        # GPU Sketch
        # Amount of work per thread, must divide D.
        # Spawn t*(D/L) instead of t*D threads when this is > 1.
        self.L = L
        assert D % L == 0
        self.DL = D // L
        
        # Use the jitclass TS to copy hashes and signs parameters.
        # This is needed, because calling random returns different random
        # numbers inside and outside of jitted functions.
        # Ideally we'd inherit from TS, but inheriting from jitted classes is
        # not possible.
        self.ts = TS(A=A,t=t,D=D,normalize=normalize)
        self.hashes = np.array(self.ts.hashes, dtype=np.int32)
        self.signs = np.array(self.ts.signs, dtype=np.float32)

        self.d_hashes = cuda.to_device(self.hashes)
        self.d_signs = cuda.to_device(self.signs)

    def sketch(self, seqs: list[nb.int8[:]]) -> list[nb.float32[:]]:
        assert isinstance(seqs, List)
        assert len(seqs) > 0

        # TODO: Add normalization to the GPU sketch method.
        for seq in seqs:
            assert (
                len(seq) ** self.t < 10 ** 38
            ), "Counts may overflow! Lower t or shorten the sequence."

        # Sort by decreasing length
        seqs = sorted(seqs, key=lambda s: len(s), reverse=True)

        # Put all operations on a stream, so that the python code runs asynchronously of the GPU code.
        stream = cuda.stream()

        # Launch one thread block per sequence.
        blocks = len(seqs)

        # Convert the input sequences to a single list of characters and the corresponding start indices.
        raw_seq = np.concatenate(seqs)
        starts = np.array(
            np.cumsum(np.array([0] + [len(seq) for seq in seqs]), dtype=np.int32),
            dtype=np.int32,
        )

        # Copy data from host to device.
        d_raw_seq = cuda.to_device(raw_seq, stream=stream)
        d_starts = cuda.to_device(starts, stream=stream)
        d_T = cuda.device_array((blocks, self.t + 1, self.D), dtype=np.float32, stream=stream)

        threads = self.t * self.D // self.L

        # Make sure we have enough threads to initialize self.hashes and
        # self.signs by a single synchronous copy.
        assert self.DL >= self.A

        # One thread per (l, k) <= (D/L, t)
        _gpu_sketch[
            (blocks, 1),
            (self.DL, self.t),
            stream,
            4 * (threads + 2 * (self.t + 1) * self.D + 2 * self.A * self.t),
        ](
            np.int32(self.A),
            np.int32(self.t),
            np.int32(self.D),
            np.int32(self.L),
            self.d_hashes,
            self.d_signs,
            d_raw_seq,
            d_starts,
            d_T,
        )

        T = d_T.copy_to_host(stream=stream)

        # Only return the length t sketch
        sketched_seqs = List()
        for seq, sketch in zip(seqs, T):
            self.ts._normalize(seq, sketch[self.t])
            sketched_seqs.append(sketch[self.t])

        return sketched_seqs

    def sketch_one(self, seq: nb.int8[:]) -> nb.float32[:]:
        return self.sketch(List([seq]))[0]
    
    def dist(self, s1, s2):
        return l2_dist(s1, s2)


class TSS():
    def __init__(self, t, W, S, D, A, seq_len, normalize = True, sketch_class=TS):
        # reduce sketch dim to ensure that the flattened sketch size is D 
        D2 = int(D/int((seq_len-W)/S)+1)   
        self.sketcher = sketch_class(t=t,D=D2,A=A,normalize=normalize)
        self.t = t
        self.W = W
        self.S = S
        self.D2 = D2
        self.D = D
    
    def sketch_one(self, seq: nb.int32[:]) -> nb.float32[:,:]:
        L = int(np.ceil((len(seq)-self.W+1)/self.S))
        sketch = np.zeros((self.D2,L), dtype=np.float32)
        for si,i in enumerate(np.arange(0,len(seq)-self.W+1,self.S)):
            sketch[:,si] = self.sketcher.sketch_one(seq[i:i+self.W])
        
        return sketch.flatten()
    
    def sketch(self, seqs):
        all_seqs = List()
        starts = np.arange(0,len(seqs[0])-self.W+1,self.S)
        n, m = len(seqs),len(starts)
        for seq in seqs:
            for i in starts:
                all_seqs.append(seq[i:i+self.W])
        all_sketches = self.sketcher.sketch(all_seqs)
        sketches = List()
        for i in range(n):
            sk = np.array([x for sk in all_sketches[i*m:(i+1) * m] for x in sk])
            sketches.append(sk)
        return sketches
    
    def dist(self, s1, s2):
        return l2_dist(s1,s2)


# numba code for weighted randint: 
# given p[0,...,n-1], where sum(p)=1, return 'i' with probability p[i]
@njit
def randint_weighted(p=nb):
    p = np.cumsum(p/np.sum(p))
    rnd = np.random.random()
    return np.nonzero(rnd<=p)[0][0]

# given sequence `seq`, alphabet `0,...,A-1`, and mutation rate `mr`, 
# return a new sequence where every index is mutated with probaility `mr`, 
# the type of mutation is selected randomly from (ins,del,sub) with `1/3` prob.
@njit
def mutate_seq(A, seq, mr):
    N = len(seq)
    y = np.random.randint(0,A,N)
    if mr==1:               # just return random 
        return y
    p=np.array([1-mr,mr/3,mr/3,mr/3])
    i, j = 0, 0
    while j < N and i < N:
        op = randint_weighted(p)
        if op==0:         # copy 
            y[j] = (seq[i])
            i += 1
            j += 1
        elif op== 1:       # insert
            y[j] = np.random.randint(A)
            j += 1
        elif op== 2:       # delete 
            i += 1
        elif op == 3:       # substitude 
            y[j] = (seq[i]+1+np.random.randint(A-1))% A
            i += 1
            j += 1
    return y

# generate sample dataset
# last two columsn mutation rate is 1  -> totally random 
def gen_seqs(N,A,num_samples, num_rates):
    rates = np.linspace(1.0/num_rates,1,num_rates)     
    rates[-2:] = rates[-1]  
    samples = np.random.randint(0,A,size=(num_samples,len(rates)+1,N),dtype=np.int8)
    for i in tqdm.tqdm(range(num_samples),total=num_samples, desc='generating seqs '):
        for j,mr in enumerate(rates):
            samples[i,j+1,:] = mutate_seq(A=A,seq=samples[i,0,:],mr=mr)
    return samples, rates 


@njit
def edit_dist(s1, s2):
    l1 = len(s1)
    l2 = len(s2)
    distances_ = np.arange(l2+1,dtype=np.int64)
    for i in range(1,l1+1):
        distances = np.empty((l2+1),dtype=np.int64)
        distances[0] = i
        for j in range(1,l2+1):
            distances[j] = min(distances[j-1]+1,
                               distances_[j]+1,
                               distances_[j-1]+(s1[i-1]!=s2[j-1]))
        distances_ = distances
    return distances[l2]


@jitclass([
           ('i', nb.int32),
           ('n', nb.int64),
           ('step', nb.int32),
           ('steps', nb.int32),
           ('bar_len', nb.int32),
           ('desc', nb.types.string),
           ('start_time', nb.float64),
           ])
class PB_parallel:
    def __init__(self, n):
        self.i = 0
        self.n = n
        self.steps = 100
        self.bar_len = 10
        self.step = (n//self.steps)
        self.desc = 'progress'
        with objmode():
            self.start_time = time.monotonic()

    def hms(self, delta):
        delta = int(delta)
        h, rem = divmod(delta, 3600)
        m, s = divmod(rem, 60)
        L = ''
        if h>0:
            L += str(h) + ':' 
        if m<=9:
            L += '0'
        L += str(m) + ':'
        if s<=9:
            L += '0'
        L += str(s) 
        
        return L 

    def inc(self):
        self.i += 1
        if (self.i % self.step == 0) or (self.i == self.n):
            cur_step = int(self.bar_len*self.i/self.n)
            perc = int(self.i*100/self.n)
            s = '\r' + self.desc + ': ' + str(perc) + '% ' 
            s += '|' + u'\u2588'*(cur_step) + ' '*(self.bar_len-cur_step)+'| '
            s += str(self.i) + '/' + str(self.n)
            with objmode():
                speed = self.i / (time.monotonic()-self.start_time)
                remain = self.hms( 1.0/speed * (self.n-self.i) )
                overall = self.hms( 1.0/ speed * self.n )
                s += ' [' + remain + '<' + overall + ', ' + str(int(speed)) + ' it/s]'
                print(s,end='')


@njit(parallel=True)
def edit_dist_pairs(pairs,seqs,dist_func):
    n = len(pairs)
    dists = np.empty(n)
    pb = PB_parallel(n)
    pb.desc = 'edit distance'
    for pi in prange(n):
        i,j = pairs[pi]
        dists[pi] = dist_func(seqs[i],seqs[j])
        pb.inc()
        
    return dists


def calc_dists(models,all_samples):
    n, m, l = len(all_samples), len(all_samples[0]), len(all_samples[0][0])
    all_seqs = np.array(all_samples,dtype=np.int8)
    all_seqs = List(all_seqs.reshape((n*m,-1)))
    results = dict()
    throughput = dict()
    trange = tqdm.tqdm(models.items(), leave=True)
    for name, sketcher in trange:
        trange.set_description('computing ' + name)
        start_time = time.monotonic()
        all_sketches = sketcher.sketch(all_seqs)
        sketches = np.array(all_sketches)
        sketches = sketches.reshape((n,m,-1))
        sk = sketches.reshape((n,m,-1))
        dists = np.empty((n,m-1))
        for i in range(n):
            for j in range(m-1):
                dists[i,j] = sketcher.dist(sk[i,0,:],sk[i,j+1,:])
            
        throughput[name] = m*n*l/1e6/(time.monotonic() - start_time)
        results[name] = dists
    return results, throughput


def plot_dists(rates, models, results, plot_log):
    plt.figure()
    for name, dists in results.items():
        dists += np.random.randn(*dists.shape)*1e-10    # break ties arbitrarily
        detect = np.mean(dists[:,:-1]>dists[:,-1][:,np.newaxis],axis=0)
        detect[detect == 0] = 1.0 / dists.shape[0]      # 0 statisticaly means <= 1/#samples
        if plot_log:
            detect = -np.log2(detect)
        plt.plot(rates[:-1], detect, label=name,marker='.')
    plt.legend()
