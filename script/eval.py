#!/usr/bin/env python3

import sys
srcdir = "/project/mchaisso_100/cmb-16/tsungyul/work/vntr/danbing-tk/script/"
sys.path.insert(0, srcdir)
import numpy as np
import pandas as pd
import vntrutils as vu
import pickle
import glob


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

def get_annotated_nvps(g_trks, g_gks, loo_gks):
    nvps_tr = {}
    nvps_ntr = {}
    for tri in range(NTR):
        ps = g_gks[tri]
        if not ps: continue
        trks = g_trks[tri]

        loo_ps = loo_gks[tri]
        loo_es = set()
        for km, bits in loo_ps.items():
            kl = km << 2
            for i in range(4):
                if bits%2:
                    e = kl + i
                    loo_es.add(e)
                bits >>= 1

        for km, bits in ps.items():
            kmr = vu.getRCkmer(km, 21)
            ckm = min(km, kmr)
            istr = ckm in trks
            kl = km << 2
            for i in range(4):
                if bits%2:
                    e = kl + i
                    if e not in loo_es:
                        if istr:
                            if tri not in nvps_tr:
                                nvps_tr[tri] = set([e])
                            else:
                                nvps_tr[tri].add(e)
                        else:
                            if tri not in nvps_ntr:
                                nvps_ntr[tri] = set([e])
                            else:
                                nvps_ntr[tri].add(e)
                bits >>= 1
        if tri in nvps_tr:
            rm = set()
            for e in nvps_tr[tri]:
                er = vu.getRCkmer(e, 22)
                if er not in nvps_tr[tri]: # e = km0 --> km1. km0 in TR but not km1. Remove e from nvps_tr.
                    rm.add(e)
            nvps_tr[tri] -= rm
            if tri in nvps_ntr:
                nvps_ntr[tri] |= rm
            else:
                nvps_ntr[tri] = rm
            if len(nvps_tr[tri]) == 0:
                nvps_tr.pop(tri)
    return nvps_tr, nvps_ntr

def get_bi_trks(loo_trks, tri):
    trks = set()
    for km in loo_trks[tri]:
        trks.add(km)
        trks.add(vu.getRCkmer(km, 21))
    return trks

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
            e1.ue = e0
        
    def prune(self, dead, e):
        # backtrack until last branching node
        pruned = set()
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
        else:
            nodex = e.c
            
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

def prune_nvps_tr(nvps_tr, loo_trks, loo_gks):
    nvps_tr_pruned = {}

    gfs = {}
    for tri, es in nvps_tr.items():
        gf = {}
        for e in es:
            pa = e >> 2
            out = 2 ** (e%4)
            if pa not in gf:
                gf[pa] = out
            else:
                gf[pa] |= out
        gfs[tri] = gf

    for tri, gf in gfs.items():
        # load TR annot
        trks = get_bi_trks(loo_trks, tri)
        ntrks = set()
        for km in loo_gks[tri]:
            if km not in trks:
                ntrks.add(km) # useful for finding TR-NTR or bubble

        # call bubbles
        trbub = {}
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
                    if pa in trbub:
                        trbub[pa].append(alive_e)
                    else:
                        trbub[pa] = [alive_e]
        print(f"\t{tri} {len(trbub)} {nsnarl} {nnve} {len(dead)}")
        if trbub:
            nvps_tr_pruned[tri] = trbub
    return nvps_tr_pruned

# computing gt novel edge indices
def get_nvps_tr_pruned_es(nvps_tr_pruned, fn):
    def k2c(k2c_, e):
        return k2c_[e] if e in k2c_ else 0

    tri2ei = {}
    es = []
    ei = 0
    for tri in range(NTR):
        if tri not in nvps_tr_pruned:
            tri2ei[tri] = ei
            continue

        trbub = nvps_tr_pruned[tri]
        for pa, ess in trbub.items():
            for es_ in ess:
                es_l = [e for e in es_]
                es += es_l
                ei += len(es_l)
        tri2ei[tri] = ei
    es = np.array(es)
    with open(fn, "wb") as f:
        pickle.dump((es,tri2ei), f)
    return es, tri2ei

def seq2h(seq, k=1):
    n = 2**(2*k)
    npb = np.zeros(n)
    for km in vu.read2kmers(seq, k, canonical=False):
        npb[km] += 1
    npb /= len(seq)
    h = 0
    for i in range(n):
        if npb[i] > 0:
            h -= npb[i]*np.log2(npb[i])
    return h

def final_eval_CV_v1(tri2es, gt_es, gt_tri2ei, mode=None):
    stats = np.zeros([NTR,4], dtype=float) # TP, FP, FN, h3
    if mode == 2:
        tris = set(tri2es[:,0].tolist()) if tri2es.size else set()
    for tri in np.arange(NTR):
        # get ground truth
        si = gt_tri2ei[tri-1] if tri else 0
        ei = gt_tri2ei[tri]
        es = gt_es[si:ei]
        gtes = set(es.tolist())
        # get novel edge calls
        if mode is None:
            if tri in tri2es:
                es = tri2es[tri]
            else:
                es = set()
        elif mode == 1: # tri2ves[tri] = (es_ar, e2c, fcmax)
            if tri in tri2es:
                es = set(tri2es[tri][0].tolist())
            else:
                es = set()
        elif mode == 2: # tribes = [[tri, alive_e], ...]
            if tri in tris:
                m = tri2es[:,0] == tri
                es = set.union(*(tri2es[m,1]))
            else:
                es = set()
        else:
            assert False
        # eval
        TP = len(es & gtes)
        FP = len(es - gtes)
        FN = len(gtes - es)
        h3 = np.mean([seq2h(vu.decodeNumericString(e, 22), 3) for e in es-gtes])
        stats[tri] = [TP, FP, FN, h3]
    TP, FP, FN = np.sum(stats[:,:3], axis=0)
    TP_sex, FP_sex, FN_sex = np.sum(stats[NTR_AUTOSOME:,:3], axis=0)
    df = pd.DataFrame(stats, columns=["TP","FP","FN","h3"])
    df.to_csv(f"{OUT_PREF}.metric.tsv", sep="\t", na_rep=".")
    return TP, FP, FN, TP_sex, FP_sex, FN_sex

def eval_tri2es(tri2es, gt_es, gt_tri2ei, beta=0.05, mode=None):
    TP, FP, FN, TP_sex, FP_sex, FN_sex = final_eval_CV_v1(tri2es, gt_es, gt_tri2ei, mode=mode)
    FDR = FP / (TP + FP)
    FNR = FN / (TP + FN) if (TP + FN) else np.nan
    Fb = (1 + beta**2)*TP / ((1 + beta**2)*TP + (beta**2)*FN + FP)
    print(f"Fb: {Fb:.5f} FDR: {FDR:.4f} FNR: {FNR:.4f} TP: {TP} FP: {FP} FN: {FN} TP_sex {TP_sex} FP_sex {FP_sex} FN_sex {FN_sex}")



LOO_PREF = sys.argv[1]
GF_PREF = sys.argv[2]
OUT_PREF = sys.argv[3]
GT_ANNOT = sys.argv[4] # f"{WD}/ground_truth/{GN}.nvps_tr_pruned.es.tri2ei.pickle"
NTR = int(sys.argv[5])
NTR_AUTOSOME = int(sys.argv[6])

#pickle_dir = f"{WD}/ground_truth"
if not glob.glob(GT_ANNOT):
    #gf_pref = f"{FULL_GF_DIR}/{GN}"
    #loo_gf_pref = f"{LOO_DIR}/pan"
    print("Loading g_trks", flush=True)
    g_trks = load_tr_kmers(f"{GF_PREF}.rawPB.tr.kmers")
    print("Loading g_gks", flush=True)
    g_gks = load_graph_kmers(f"{GF_PREF}.rawPB.graph.kmers")
    print("Loading loo_trks", flush=True)
    loo_trks = load_tr_kmers(f"{LOO_PREF}.tr.kmers")
    print("Loading loo_gks", flush=True)
    loo_gks = load_graph_kmers(f"{LOO_PREF}.graph.kmers")
    print("Computing nvps_tr", flush=True)
    nvps_tr, _ = get_annotated_nvps(g_trks, g_gks, loo_gks)
    print("Computing nvps_tr_pruning", flush=True)
    nvps_tr_pruned = prune_nvps_tr(nvps_tr, loo_trks, loo_gks)
    print("Computing/indexing gt_es", flush=True)
    gt_es, gt_tri2ei = get_nvps_tr_pruned_es(nvps_tr_pruned, GT_ANNOT)
else:
    print("Loading gt_es", flush=True)
    with open(GT_ANNOT, "rb") as f:
        gt_es, gt_tri2ei = pickle.load(f)

print("Loading callsets", flush=True)
with open(f"{OUT_PREF}.rarevar.pickle", "rb") as f:
    tri2ves, tribes, tri2vbes = pickle.load(f)
eval_tri2es(tri2ves, gt_es, gt_tri2ei, mode=1)
eval_tri2es(tribes, gt_es, gt_tri2ei, mode=2)
eval_tri2es(tri2vbes, gt_es, gt_tri2ei)
