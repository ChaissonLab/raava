#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn import svm
import sys
srcdir = "/project/mchaisso_100/cmb-16/tsungyul/work/vntr/danbing-tk/script/"
sys.path.insert(0, srcdir)
import vntrutils as vu
import pickle



def parse_fas_his(args):
    nh = int(args[0])
    fas = args[1 : 1+nh]
    his = [int(v) for v in args[1+nh : 1+2*nh]]
    return fas, his

def g2es(gf):
    es = set()
    for km, bits in gf.items():
        kl = km << 2
        for i in range(4):
            if bits%2:
                e = kl + i
                es.add(e)
            bits >>= 1
    return es

def load_graph_kmers(fn):
    gks = np.empty(NTR, dtype=object)
    with open(fn) as f:
        tri = -1
        for line in f:
            if line[0] == ">":
                if tri >= 0:
                    gks[tri] = ks
                tri += 1
                ks = {}
            else:
                km, ct = [int(v) for v in line.split()]
                ks[km] = ct
        else:
            gks[tri] = ks
    return gks

def load_tr_kmers(fn):
    trks = np.empty(NTR, dtype=object)
    with open(fn) as f:
        tri = -1
        for line in f:
            if line[0] == ">":
                if tri >= 0:
                    trks[tri] = ks
                tri += 1
                ks = set()
            else:
                km = int(line.split()[0])
                ks.add(km)
        else:
            trks[tri] = ks
    return trks

def load_trseqs(fas, his):
    FS = 500
    omap = pd.read_csv(FN_OMAP, sep="\t", na_values=".", usecols=his, index_col=None, header=None).to_numpy()
    trseqs = np.full(omap.shape, None, dtype=object)
    for i, fa in enumerate(fas):
        ptris = np.nonzero(np.isfinite(omap[:,i]))[0]
        with open(fa) as f:
            gtri = -1
            for line in f:
                if line[0] == ">":
                    gtri += 1
                    ptri = ptris[gtri]
                else:
                    assert len(line) - 1 > 2*FS
                    trseqs[ptri,i] = line.rstrip()[FS:-FS]
    return trseqs

# not bidirectional
def get_directed_nv_ps(seqs, loo_gks):
    FS = 0 # flanks already removed in seqs
    INVALID = np.uint64(-1)
    # novel paths
    # paths from the same locus can overlap or be the same
    nv_ps = np.full(seqs.shape, None, dtype=object) 
    for tri in range(NTR):
        for h in [0,1]:
            if seqs[tri,h] is None: continue
            
            es = vu.read2kmers_noshift(seqs[tri,h], 22, leftflank=FS, rightflank=FS, canonical=False)
            loo_es = g2es(loo_gks[tri])
            nvp = []
            for e in es:
                assert INVALID not in loo_es
                if e not in loo_es and e != INVALID:
                    nvp.append(e)
                else:
                    if nvp:
                        if nv_ps[tri,h] is None:
                            nv_ps[tri,h] = [nvp]
                        else:
                            nv_ps[tri,h].append(nvp)
                        nvp = []
    return nv_ps

# stats not bidirectional;  tri2es bidirectional
def get_directed_gt_tri2nves_stats_v1(nv_ps):
    n0 = np.sum(np.any(nv_ps != None, axis=1))
    print(f"{n0} loci with novel paths")

    n_p = np.zeros(NTR, dtype=int) # num of paths
    n_up = np.zeros(NTR, dtype=int) # num of unique paths
    n_e = np.zeros(NTR, dtype=int) # num of unique edges
    alc = {} # allele count {tri:{al:[c0,c1], ...}, ...}  ### assuming no partial overlap between paths
    n_homp = 0 # homopolymer paths
    homps = ["A"*22, "C"*22, "G"*22, "T"*22]
    n_povl = 0 # num of partial overlapping paths
    tri2es = {}

    for tri in range(NTR):
        # p: a path
        # nvps: all paths in a hap in a locus
        # cps: canonical path string set of a locus
        # es_: edge set of a path
        # es: edge set of all paths in a locus 
        # als: [(es_, p), ...]
        # alc_: allele count
        
        cps = set()
        es = set()
        als = []
        alc_ = {}
        for h in [0,1]:
            nvps = nv_ps[tri,h]
            if nvps is None: continue

            n_p[tri] += len(nvps)
            for p in nvps:
                es_ = set(p)
                estr = vu.decodeNumericString(int(p[0]), 22)
                if estr in homps and len(es_) == 1: # whole path is a homopolymer run
                    n_homp += 1
                    continue

                es |= es_
                pr = [vu.getRCkmer(int(e), 22) for e in p[::-1]]
                ers_ = set(pr)
                pstr = ".".join([str(e) for e in p]) # path string
                prstr = ".".join([str(e) for e in pr])
                if pstr < prstr:
                    ces_ = es_ # canonical edge set
                    cpstr = pstr # canonical path string
                else:
                    ces_ = ers_
                    cpstr = prstr

                # check independence of each path
                if cpstr not in cps:
                    for al_s, al_l in als:
                        ovl = al_s & ces_
                        if len(ovl): # partial overlap
                            n_povl += 1
                            print(f"\t! {tri}.{h} path size {(len(p),len(ces_))} partially overlaps (n={len(ovl)}) with allele size {(len(al_l),len(al_s))}. e.g: {[vu.decodeNumericString(int(e), 22) for e in ovl][0]}")
                            break
                    alc_[cpstr] = [0,0]
                    als.append([ces_, p]) # allele_as_set, allelle_as_list
                    cps.add(cpstr)
                alc_[cpstr][h] += 1
        n_up[tri] = len(cps)
        n_e[tri] = len(es)
        alc[tri] = alc_
        if es:
            tri2es[tri] = es | set([vu.getRCkmer(int(e), 22) for e in es])

    n1 = np.sum(n_p)
    n2 = np.sum(n_up)
    n3 = np.sum(n_e)
    n3a = np.sum([len(v) for v in tri2es.values()])
    n4 = sum([len(v) for v in alc.values()])
    n5 = np.sum([bool(v1h0 & v1h1) for v0 in alc.values() for v1h0, v1h1 in v0.values()])
    n6 = np.sum([not bool(v1h0 & v1h1) for v0 in alc.values() for v1h0, v1h1 in v0.values()])
    alcopies = [v1h0+v1h1 for v0 in alc.values() for v1h0, v1h1 in v0.values()]
    s0 = np.mean(alcopies)
    s1 = min(alcopies)
    s2 = max(alcopies)
    print(f"{n_homp} homopolymer paths removed from analysis")
    print(f"{len(tri2es)} loci with novel paths after cleaning")
    print(f"{n1} paths")
    print(f"{n_povl} partial overlapping paths")
    print(f"{n2} unique canonical paths")
    print(f"{n3} unique edges")
    print(f"{n3a} unique edges, including reverse-complement")
    print(f"Treating each unique path as an allele (n={n4})")
    print(f"{n5} HOM, {n6} HET alleles")
    print(f"mean allele copies {s0:.1f}, min {s1}, max {s2}")

    return (n_p, n_up, n_e, alc), tri2es

def es2bigf(es, k=22, bi=True):
    gf = {}
    for e in es:
        pa = e >> 2
        nt = e % 4
        if pa not in gf:
            gf[pa] = 2**nt
        else:
            gf[pa] |= 2**nt
        # make it bidirectional
        if bi:
            er = vu.getRCkmer(e, k)
            par = er >> 2
            ntr = er % 4
            if par not in gf:
                gf[par] = 2**ntr
            else:
                gf[par] |= 2**ntr
    return gf

def get_bi_trks(loo_trks, tri):
    trks = set()
    for km in loo_trks[tri]:
        trks.add(km)
        trks.add(vu.getRCkmer(km, 21))
    return trks

class Edge:
    def __init__(self, edge, parent, child):
        self.e = edge
        self.p = parent
        self.c = child
        self.a = False # isalive
        self.ue = None # upstream edge
        self.de = [] # downstream edge(s)
    def __str__(self):
        return f"{self.e} {self.p} {self.c} {self.a} up: {self.ue.e if self.ue else None} down: {[e.e for e in self.de]}"

class Cyclic_DFS:
    def __init__(self):
        self.q = [] # queue
        self.g = set() # growing nodes
        self.sni2nx = [] # [(nodex0, edgex0), ...]
        self.sni2n = [] # [set([node0, ...]), ...]
        self.sni2e = [] # [[e0, ...], ...]
        self.n2sni = {} # {node0:supernode_id, ...}
        
    def add(self, e0, e1s):
        for e1 in e1s:
            e0.de.append(e1)
            # assert e1.ue is None
            e1.ue = e0
        
    def prune(self, dead, e):
        # backtrack until last branching node
        pruned = set()
        # assert len(e.de) < 2 and e.e is not None
        while len(e.de) < 2 and e.e is not None:
            pruned.add(e.c)
            e_ = e
            e = e.ue
        if e.e is not None: # not the root edge
            e.de.remove(e_)
            e_.ue = None
        dead |= pruned
        self.g -= pruned
        return e
            
    def remove_supernode(self, sni):
        # assert sni == len(self.sni2n)-1, print(f"sni {sni} != len(self.sni2n)-1 {len(self.sni2n)-1}")
        for n in self.sni2n[sni]:
            self.n2sni.pop(n)
        self.sni2nx.pop(sni)
        self.sni2n.pop(sni)
        self.sni2e.pop(sni)
    
    def make_alive(self, alive, alive_e, e):
        # bacaktrack until an alive edge
        survived = set()
        while True:
            if e.e is None: break # root edge
            if e.a: break
            if e.p in self.n2sni: # pa is in a supernode
                sni = self.n2sni[e.p]
                nodex, edgex = self.sni2nx[sni]
                survived |= self.sni2n[sni]
                for e_ in self.sni2e[sni]:
                    alive_e.add(e_.e)
                    e.a = True
                self.remove_supernode(sni)
                e = edgex
            else:
                survived.add(e.p)
                alive_e.add(e.e)
                e.a = True
                e = e.ue
        alive |= survived
        self.g -= survived
        return self.q[-1].ue if self.q else None
        
    def merge(self, e):
        if e.c in self.n2sni:
            sni = self.n2sni[e.c]
            nodex, _ = self.sni2nx[sni]
            # print("complex", nodex, vu.decodeNumericString(nodex, 21), e, vu.decodeNumericString(e.e, 22), sni)
        else:
            nodex = e.c
            # print("simple", nodex, vu.decodeNumericString(nodex, 21), e, vu.decodeNumericString(e.e, 22))
            
        # backtrack until nodex
        sn = set([e.p, e.c])
        se = [e]
        usni = set([self.n2sni[e.p]]) if e.p in self.n2sni else set()
        npa = self.q[-1].p if self.q else None # next pa to start dfs
        found = e if e.c == npa else False
        while e.p != nodex:            
            e = e.ue
            if e.e is None: assert False
            if e.c == npa:
                found = e
            if e.p in self.n2sni:
                sni = self.n2sni[e.p]
                usni.add(sni)
            else:
                sn.add(e.p)
                se.append(e)
        
        if usni:
            for sni in usni:
                sn |= self.sni2n[sni]
                se += self.sni2e[sni]
                self.sni2nx[sni] = None
                self.sni2n[sni] = None
                self.sni2e[sni] = None
        self.sni2nx.append((nodex, e.ue))
        self.sni2n.append(sn)
        self.sni2e.append(se)
        sni = len(self.sni2nx) - 1
        for n in sn:
            self.n2sni[n] = sni
        
        return found if found else e
            
    def check_survival(self, dead, e0):
        ch = e0.c
        if ch not in self.n2sni: return None
    
        sni = self.n2sni[ch]
        nodex, _ = self.sni2nx[sni]
        if ch != nodex: return None
    
        e1s = e0.de
        isalive = any([e1.a for e1 in e1s])
        # assert not isalive, print(f"make_alive did not remove nodex before check_survival")
        # assert sni == len(self.sni2n)-1, print(f"sni {sni} != len(self.sni2n)-1 {len(self.sni2n)-1}")
        e0.de = []
        for e1 in e1s:
            e1.ue = None
        ns = self.sni2n[sni]
        dead |= ns
        self.g -= ns
        self.remove_supernode(sni)
        return self.prune(dead, e0)

def check_edge_v1(gf, trks, ntrks, e, dfs, alive, alive_e, dead, verbose=False):
    """
    return: isalive, bte
        is_alive:
            0: dead
            1: growing, non-terminal
            2: growing, terminal, merged with existing growing branch 
            3: alive
        bte
            - backtrack edge, used to traverse upstream in search for dfs.q[-1].ue
            - if bte == 0:    dfs.q is empty
            - if bte is None: growing path, no need to backtrack
    """
    bte = [None]
    if e.p == e.c: # when it forms a self-loop
        if verbose: print("[X.homo]",end=" ")
        bte = dfs.prune(dead, e)
        return 0, bte
    
    if e.c in alive: # when it merges with an alive branch
        if verbose: print("[O.merge]", end=" ")
        bte = dfs.make_alive(alive, alive_e, e)
        return 3, bte
    if e.c in trks: # complete bubble
        if verbose: print("[O.tr]", end=" ")
        bte = dfs.make_alive(alive, alive_e, e)
        return 3, bte

    if e.c not in gf: # when it's a tip
        if verbose: print(f"[X.tip]",end=" ")
        dead.add(e.c)
        # assert e.c not in trks
        bte = dfs.prune(dead, e)
        return 0, bte
    if e.c in dead: # when it merges with a dead branch
        if verbose: print("[X.dead]",end=" ")
        bte = dfs.prune(dead, e)
        return 0, bte
    if e.c in ntrks: # when it reaches NTR
        if verbose: print("[X.NTR]",end=" ")
        bte = dfs.prune(dead, e)
        return 0, bte
    
    if e.c in dfs.g: # when it merges with a growing branch
        if verbose: print("[m.grow]",end=" ")
        bte = dfs.merge(e)
        return 2, bte
    else: # growing branch w/ unknown survival
        dfs.g.add(e.c)
        return 1, 0
    
def decode_edges(gf, pa):
    out = gf[pa]
    es = []
    mask = (1<<(2*21)) - 1
    pa_km1 = ((pa << 2) & mask)
    for i in range(4):
        if out % 2:
            ch = pa_km1 + i
            e = (pa << 2) + i
            es.append(Edge(e, pa, ch))
        out >>= 1
    ne = len(es)
    return ne, es

def check_bubble_root_edge(rt, edge, gf, trks, ntrks, alive, dead):
    alive_e = set()
    dfs = Cyclic_DFS()
    dfs.q = [edge]
    dfs.add(rt, [edge])
    while True:
        e0 = dfs.q.pop()
        isalive, bte = check_edge_v1(gf, trks, ntrks, e0, dfs, alive, alive_e, dead)
        while bte == 0: # growing path, no need to backtrack
            ne, e1s = decode_edges(gf, e0.c)
            dfs.add(e0, e1s)
            if ne > 1:
                for i in range(len(e1s)-1):
                    dfs.q.append(e1s[i])
            e0 = e1s[-1]
            isalive, bte = check_edge_v1(gf, trks, ntrks, e0, dfs, alive, alive_e, dead)

        # backtrack till dfs.q[-1].ue
        if not dfs.q: break            
        npa = dfs.q[-1].p # next pa to start dfs
        while bte.c != npa: # done traversing the subtree of bte
            out = dfs.check_survival(dead, bte) # check nodex and survival
            if out is None:
                bte = bte.ue
            else:
                bte = out
    return alive_e

def es2bes(tri2es, loo_gks, loo_trks, CTH=10):
    def k2c(k2c_, e):
        return k2c_[e] if e in k2c_ else 0
    
    tri2bes = {}
    for tri, nves in tri2es.items():
        gf = es2bigf([int(e) for e in nves], k=22, bi=False)
        
        # load TR annot
        trks = get_bi_trks(loo_trks, tri)
        ntrks = set()
        for km in loo_gks[tri]:
            if km not in trks:
                ntrks.add(km) # useful for finding TR-NTR or bubble
        
        # call bubbles
        alive, dead = set(), set()
        nsnarl, nnve = 0, 0 # snarl, novel_edge
        for pa in gf:
            if pa not in trks: continue

            rt = Edge(None, None, pa) # root edge
            ne, es = decode_edges(gf, pa)
            for edge in es:
                alive_e = check_bubble_root_edge(rt, edge, gf, trks, ntrks, alive, dead)
                if alive_e:
                    nsnarl += 1
                    nnve += len(alive_e)
                    if tri in tri2bes:
                        tri2bes[tri] |= alive_e
                    else:
                        tri2bes[tri] = alive_e
        if len(tri2bes) < 5:
            print(f"{tri} {nsnarl} {len(tri2bes[tri]) if nnve else 0} {nnve} {len(dead)}")
    return tri2bes

# ground truth es is bidirectional
def save_ground_truth_TR_bubble_es_tri2ei(tri2bes, FOUT):
    tri2ei = {}
    es = []
    ei = 0
    for tri in range(NTR):
        if tri not in tri2bes:
            tri2ei[tri] = ei
            continue

        es_l = list(tri2bes[tri])
        es += es_l
        ei += len(es_l)
        tri2ei[tri] = ei
    es = np.array(es).astype(int)
    print(".", end="")
    with open(FOUT, "wb") as f:
        pickle.dump((es,tri2ei), f)
    print(".")

NTR = int(sys.argv[1])
FN_OMAP = sys.argv[2]
LOO_PREF = sys.argv[3]
FOUT = sys.argv[4]
fas_, his_ = parse_fas_his(sys.argv[5:])

print("Loading LOO graph.kmers", flush=True)
loo_gks_ = load_graph_kmers(f"{LOO_PREF}.graph.kmers")
print("Loading LOO tr.kmers", flush=True)
loo_trks_ = load_tr_kmers(f"{LOO_PREF}.tr.kmers")
print("Loading trseqs", flush=True)
trseqs_ = load_trseqs(fas_, his_)
print("Computing nv_ps", flush=True)
nv_ps_ = get_directed_nv_ps(trseqs_, loo_gks_)
print("Computing tri2nves stats", flush=True)
path_stats, gt_tri2nves_ = get_directed_gt_tri2nves_stats_v1(nv_ps_)
print("Computing bidirected tri2nbes", flush=True)
gt_tri2nbes_ = es2bes(gt_tri2nves_, loo_gks_, loo_trks_)
print("Saving ground truth", flush=True)
save_ground_truth_TR_bubble_es_tri2ei(gt_tri2nbes_, FOUT)
