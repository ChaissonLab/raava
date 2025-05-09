#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import pickle
from sklearn import svm
from TR_bubble_search import read2kmers, decodeNumericString, getRCkmer, e2ce, k2ck, Edge, decode_edges, es2bigf, check_bubble_root_edge


def load_precision_filter(fn, rc_th=0.9):
    df = pd.read_csv(fn, sep="\t")
    ps = df["Precision"]
    rc = df["Recall"]
    return (ps == 1) & (rc >= rc_th)

def get_tri2ks(fn, add_rv=False):
    tri2ks = np.empty(NTR, dtype=object)
    with open(fn) as f:
        tri = -1
        for line in f:
            if line[0] == ">":
                if tri >= 0:
                    tri2ks[tri] = ks
                tri += 1
                if tri >= NTR_AUTOSOME and SEX == 1:
                    tri2ks[tri] = ks
                    break
                ks = set()
            else:
                km = int(line.split()[0])
                ks.add(km)
                if add_rv:
                    ks.add(getRCkmer(km, 21))
        else:
            tri2ks[tri] = ks
    return tri2ks

def load_tr_kmc(fn, index):
    # return: canonical trkmer -> count
    tr2trk2c = {}
    f0 = open(index)
    f1 = open(fn)
    tri = -1
    for line in f0:
        if line[0] == ">":
            tri += 1
            if tri >= NTR_AUTOSOME and SEX == 1: break
            tr2trk2c[tri] = {}
        else:
            km = int(line.split()[0])
            ct = int(f1.readline().rstrip())
            tr2trk2c[tri][km] = ct
    f0.close()
    f1.close()
    return tr2trk2c

def load_bubbles(fn):
    bub_kmc = {}
    with open(fn) as f:
        tri = -1
        for line in f:
            if line[0] == ">":
                if tri >= 0:
                    bub_kmc[tri] = k2c
                tri += 1
                if tri >= NTR_AUTOSOME and SEX == 1: break
                k2c = {}
            else:
                km, ct = [int(v) for v in line.split()]
                ce = e2ce(km)
                if ce in k2c:
                    k2c[ce] += ct
                else:
                    k2c[ce] = ct
        else:
            bub_kmc[tri] = k2c
    return bub_kmc

def seq2h(seq, k=1):
    n = 2**(2*k)
    npb = np.zeros(n)
    for km in read2kmers(seq, k, canonical=False):
        npb[km] += 1
    npb /= len(seq)
    h = 0
    for i in range(n):
        if npb[i] > 0:
            h -= npb[i]*np.log2(npb[i])
    return h

class BubbleRoot:
    def __init__(self):
        self.crt = [] # c_root
        self.cne = [] # c_novel_edge
        self.cee = [] # c_existing_edge(s)
        self.cte = [] # c_trimmed_edge(s)
        self.rnr = [] # c_novel_edge/c_root
        self.rne = [] # c_novel_edge/c_existing_e
        self.ncb = [] # n_snarl(ComplexBubble)
        self.ntm = [] # n_trimmed_edge
        self.nbe = [] # n_bubble_edge
        self.h1 = []  # entropy (k=1)
        self.h2 = []  # entropy (k=2)
        self.h3 = []  # entropy (k=3)
        self.tv = []  # true variant (the novel edge only; the whole snarl could be only partially true)

        self.es = []      # all edges from bubble
        self.cts = []     # corresponding count for each edge
        self.bi2ei = []   # bubble ind -> edge end ind
        self.tri2bei = np.zeros(NTR, dtype=int) # tri -> bubble end ind

def find_TR_snarls(qcfilter, tri2trks, tri2ntrks, bub_tr2k2c, tr_kmc, TH_CNE=10, verbose=False):
    br = BubbleRoot()
    bi = 0 # bubble ind
    for tri in bub_tr2k2c:
        if not qcfilter[tri] or len(bub_tr2k2c[tri]) > GRAPHSIZELIMIT:
            br.tri2bei[tri] = len(br.cne)
            continue

        # load TR annot
        trks = tri2trks[tri]
        ntrks = tri2ntrks[tri]

        # load novel edges
        bk2c = bub_tr2k2c[tri]
        tk2c = tr_kmc[tri]
        kms = bk2c.keys()

        # build graph
        gf = es2bigf(kms)

        # find TR bubble
        alive, dead = set(), set()
        alive_es = set()
        naes = []

        for pa in gf:
            if pa not in trks: continue

            crt = tk2c[k2ck(pa)]
            cee = crt
            cte = 0
            ncb = 0
            ntm = 0
            rt = Edge(None, None, pa) # root edge
            ne, edges = decode_edges(gf, pa)
            for edge in edges:
                cne = bk2c[e2ce(edge.e)]
                if cne < TH_CNE: # prefilter low-coverage novel edge
                    cee -= cne
                    cte += cne
                    ntm += 1
                    continue

                alive_e = check_bubble_root_edge(rt, edge, gf, trks, ntrks, alive, dead)
                naes.append(len(alive_e))
                alive_es |= alive_e

                e_ = edge.e
                if edge.a:
                    cee -= cne
                    seq = decodeNumericString(e_, 22)
                    ncb += 1
                    br.crt.append(crt)
                    br.cne.append(cne)
                    br.rnr.append(cne / crt)
                    br.nbe.append(naes[-1])
                    br.h1.append(seq2h(seq, k=1))
                    br.h2.append(seq2h(seq, k=2))
                    br.h3.append(seq2h(seq, k=3))

                    es = [e_] + [e for e in alive_e if e != e_]
                    bi += 1
                    br.es += es
                    br.cts += [bk2c[e2ce(e)] for e in es]
                    br.bi2ei.append(len(br.es))
                else:
                    cte_ = bk2c[e2ce(e_)]
                    cee -= cte_
                    cte += cte_
                    ntm += 1
            if ncb:
                for i in range(ncb):
                    br.cee.append(cee)
                    br.cte.append(cte)
                    br.rne.append(br.cne[-ncb+i] / (cee+1))
                    br.ncb.append(ncb)
                    br.ntm.append(ntm)
        if alive and verbose:
            print(f"\t{tri} {len(alive)} {len(dead)} {len(alive_es)} {len(naes)} {np.median(naes)} {np.mean(naes):.1f}")

        br.tri2bei[tri] = len(br.cne)
    return br


# export statistics matrix; col: feature; row: snarl
def cov_norm_br_svm(br, fn, TH):
    df = pd.DataFrame()
    df["c_root"] = br.crt
    df["c_nv_e"] = br.cne
    df["c_ex_e"] = br.cee
    df["c_tm_e"] = br.cte
    df["r_nve_r"] = br.rnr
    df["r_nve_exe"] = br.rne
    df["n_snrl"] = br.ncb
    df["n_tm_e"] = br.ntm
    df["n_bb_e"] = br.nbe
    df["h1"] = br.h1
    df["h2"] = br.h2
    df["h3"] = br.h3
    #df.astype({"c_root": float, "c_nv_e": float, "c_ex_e": float, "c_tm_e": float})
    #qt999 = np.quantile(df["c_nv_e"], 0.999) # quantile 0.999 = 15 in hs1 benchmark
    #FC = (15/qt999) # fold change
    #df["c_root"] *= FC
    #df["c_nv_e"] *= FC
    #df["c_ex_e"] *= FC
    #df["c_tm_e"] *= FC

    br.cts = np.array(br.cts) #* FC

    m0 = (df["c_nv_e"] > TH).to_numpy()
    X = df[m0]
    bis = np.nonzero(m0)[0] # bubble indices
    with open(fn, "rb") as f:
        clf = pickle.load(f)
    yh = clf.predict(X) # y hat
    return df, bis[yh] # feature_mat, valid_bubble_indices

# filter bubbles based on the br-SVM model + heauristic min cov filtering on bubble edges
def filter_bubble_edges(vbis, br, TH1, TH2, TH3, verbose=False):
    es, cts, bi2ei, tri2bei = br.es, br.cts, br.bi2ei, br.tri2bei
    vbis_s = set(vbis.tolist())
    tri2ves = {}
    for tri in range(NTR):
        # ives = set() # invalid edges
        es_ar = np.array([], dtype=np.int64)
        ct_ar = np.array([], dtype=np.int64)
        tcmin = 99999
        fcmax = 0
        valid_locus = False # at least one valid bubble
        bsi = tri2bei[tri-1] if tri else 0
        bei = tri2bei[tri]
        for bi in range(bsi, bei):
            esi = bi2ei[bi-1] if bi else 0
            eei = bi2ei[bi]
            assert eei != esi
            c = cts[esi]
            if bi in vbis_s:
                valid_locus = True
                tcmin = min(tcmin, c)
                es_ar = np.concatenate((es_ar, es[esi:eei]))
                ct_ar = np.concatenate((ct_ar, cts[esi:eei]))
            else:
                fcmax = max(fcmax, c)
        if not valid_locus: # all bubbles are removed by SVM
            continue

        # heuristic cov filtering
        if tcmin-fcmax > TH3:
            TH = max(TH2, fcmax)
        else:
            TH = max(TH2, tcmin-TH1)

        assert len(set(es_ar.tolist())) == es_ar.size
        mask = ct_ar > TH
        es_ar = es_ar[mask]
        ct_ar = ct_ar[mask]
        e2c = {}
        for e_f, c in zip(es_ar, ct_ar):
            for e in [e_f, getRCkmer(e_f, 22)]:
                if e in e2c:
                    assert e2c[e] == c
                else:
                    e2c[e] = c

        tri2ves[tri] = (es_ar, e2c, fcmax)
    if verbose: print(np.sum([v[0].size for v in tri2ves.values()]), "edges")
    return tri2ves

# run snarl calling again to refine; compute bubble path features
def get_bubble_path_features(tri2ves, tri2trks, tri2ntrks, verbose=False):
    def count_ngap(cs):
        n3gp = 0
        n4gp = 0
        for i, c in enumerate(cs):
            if i == 0:
                c_ = c
            else:
                gp = abs(c-c_)
                if gp >= 3:
                    n3gp += 1
                    if gp >= 4:
                        n4gp += 1
                c_ = c
        return n3gp, n4gp

    def fill_bubble_path_bidirectionality(es, bres, bdf):
        j = -len(bres)
        for i in range(len(bres)):
            bdf[j][7] = getRCkmer(bres[i],22) in es
            j += 1

    tribes = []
    tri2bes = {}
    bdf = []
    for tri in sorted(tri2ves.keys()):
        es_ar, e2c, fcmax = tri2ves[tri]

        # load TR annot
        trks = tri2trks[tri]
        ntrks = tri2ntrks[tri]

        # convert kms (k+1)-mer to graph
        gf = es2bigf(es_ar, bi=False)

        # find TR bubble
        alive, dead = set(), set()
        alive_es = set() #[]
        bres = [] # bubble root edge; used to check bidirectionality

        for pa in gf:
            if pa not in trks: continue

            rt = Edge(None, None, pa) # root edge
            ne, edges = decode_edges(gf, pa)
            for edge in edges:
                alive_e = check_bubble_root_edge(rt, edge, gf, trks, ntrks, alive, dead)

                if alive_e:
                    cs = [e2c[e] for e in alive_e]
                    bbs, mcov, scov = len(alive_e), np.mean(cs), np.std(cs)
                    n3gp, n4gp = count_ngap(cs)
                    bdf.append([bbs, mcov, scov, fcmax, mcov-fcmax, n3gp, n4gp, None])
                    bres.append(edge.e)
                    tribes.append([tri, alive_e])
                    alive_es |= alive_e
        if alive_es:
            fill_bubble_path_bidirectionality(alive_es, bres, bdf)

    # feature mat for bubble path:
    #   bbs: bubble size
    #   mcov: mean(cov)
    #   scov: std(cov)
    #   fcmax: (False bubble root Cov).MAX
    #   cdiff: Cov DIFF = tcmin - fcmax
    #   n3gp: num of 3-gap. Num of N-gap (cov diff btwn pa/ch edges >= N)
    #   n4gp: num of 4-gap
    #   bdir: bidirectionality (if the reverse complement exists in ground truth)
    # label: True if any edge is in ground truth
    bdf = pd.DataFrame(bdf, columns=["bbs","mcov","scov","fcmax","cdiff","n3gp","n4gp","bdir"])
    if verbose: print(np.sum([len(v) for _, v in tribes]), "edges")
    return np.array(tribes, ndmin=2), bdf

def get_valid_bubble_edges(tribes, bdf, fn):
    with open(fn, "rb") as f:
        clf = pickle.load(f)
    yh = clf.predict(bdf)
    tri2vbes = {}
    for tri, bes in tribes[yh]:
        if tri not in tri2vbes:
            tri2vbes[tri] = bes
        else:
            tri2vbes[tri] |= bes
    ne = 0
    for tri, vbes in tri2vbes.items():
        rves = set([getRCkmer(e, 22) for e in vbes])
        tri2vbes[tri] |= rves
        ne_ = len(tri2vbes[tri])
        ne += ne_
        print(f"tri: {tri} has {ne_} edges (bidirection)")
    print(f"{ne} edges called in total", flush=True)
    return tri2vbes






LOO_DIR = sys.argv[1]
GT_DIR = sys.argv[2]
WD = sys.argv[3]
GN = sys.argv[4]
FILTER = sys.argv[5]
BR_SVM = sys.argv[6]
BP_SVM = sys.argv[7]
NTR = int(sys.argv[8])
NTR_AUTOSOME = int(sys.argv[9])
SEX = int(sys.argv[10])
TH = int(sys.argv[11])
TH1 = int(sys.argv[12])
TH2 = int(sys.argv[13])
TH3 = int(sys.argv[14])
GRAPHSIZELIMIT = int(sys.argv[15])

TRKS = f"{LOO_DIR}/pan.tr.kmers"
NTRKS = f"{LOO_DIR}/pan.graph.kmers"
TRINDEX = f"{LOO_DIR}/pan.reindex.tr.kmers"

print(f"Loading qcfilter {FILTER}", flush=True)
qcfilter = load_precision_filter(FILTER, rc_th=1)

print(f"Loading tri2trks {TRKS}", flush=True)
tri2trks = get_tri2ks(TRKS, add_rv=True)

print(f"Loading tri2ntrks {NTRKS}", flush=True)
tri2ntrks = get_tri2ks(NTRKS, add_rv=False)

fn = f"{GT_DIR}/{GN}.tr.kmers"
print(f"Loading tr_kmc {fn}\t{TRINDEX}", flush=True)
tr_kmc = load_tr_kmc(fn, TRINDEX)

fn = f"{GT_DIR}/{GN}.bub"
print(f"Loading bub_kmc {fn}", flush=True)
bub_kmc = load_bubbles(fn)

print("1st snarl finding", flush=True)
br = find_TR_snarls(qcfilter, tri2trks, tri2ntrks, bub_kmc, tr_kmc, TH_CNE=TH, verbose=True)

print("Bubble root SVM filtering", flush=True)
feature_mat, vbis = cov_norm_br_svm(br, BR_SVM, TH)
feature_mat.to_csv(f"{WD}/{GN}.br_mat.csv", sep="\t")
print(feature_mat.shape, flush=True)

print("Heuristic filtering", flush=True)
tri2ves = filter_bubble_edges(vbis, br, TH1=TH1, TH2=TH2, TH3=TH3)

print("2nd snarl finding", flush=True)
tribes, bdf = get_bubble_path_features(tri2ves, tri2trks, tri2ntrks)

print("Bubble path SVM filtering")
tri2vbes = get_valid_bubble_edges(tribes, bdf, BP_SVM)

with open(f"{WD}/{GN}.rarevar.pickle", "wb") as f:
    pickle.dump((tri2ves, tribes, tri2vbes), f)
