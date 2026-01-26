# aaRS Research Network Configuration

**Created:** 2026-01-17
**Status:** Building (8 jobs running)

## Overview

Multi-network RefNet build covering the complete aaRS research landscape.
Organized into 9 topic-focused networks that will be merged.

## Networks

| # | Network | Location | Seeds | Focus |
|---|---------|----------|-------|-------|
| 1 | **aaRS Urzymes** | `/tmp/aars_fixed/` | Carter lab papers | Evolution, minimal enzymes |
| 2 | **ProRS** | `/tmp/prorsrun/` | Halofuginone, EPRS | Prolyl-tRNA synthetase (main focus) |
| 3 | **ThrRS** | `/tmp/thrrsrun/` | Zinc binding, editing | Threonyl-tRNA synthetase (main focus) |
| 4 | **aaRS Family** | `/tmp/aars_family/` | Class I/II partition | Structural classification |
| 5 | **Ancestral/LUCA** | `/tmp/ancestral_luca/` | Pre-LUCA, genetic code origin | Deep evolution |
| 6 | **Disease** | `/tmp/aars_disease/` | CMT, neurodegeneration | Clinical/pathology |
| 7 | **Non-canonical** | `/tmp/aars_noncanonical/` | mTOR, angiogenesis | Signaling functions |
| 8 | **MSC** | `/tmp/aars_msc/` | Multi-synthetase complex | Complex assembly |
| 9 | **Promiscuity** | `/tmp/aars_promiscuity/` | Misacylation, editing | Substrate selectivity |
| 10 | **tRNA Identity** | `/tmp/aars_trna_identity/` | Recognition elements | tRNA specificity |
| 11 | **Mitochondrial** | `/tmp/aars_mitochondrial/` | mt-aaRS, organellar | Organellar specialization |
| 12 | **Code Expansion** | `/tmp/aars_code_expansion/` | PylRS, unnatural AA | Synthetic biology |
| 13 | **Inhibitors** | `/tmp/aars_inhibitors/` | Mupirocin, drug design | Drug discovery |

## Seed DOIs

### 1. aaRS Urzymes (Carter lab focus)
- `10.1074/jbc.m113.496125` - Aminoacylating Urzymes Challenge RNA World
- `10.1007/s00239-015-9672-1` - Ancestral Pre-LUCA aaRS
- `10.1261/rna.061069.117` - aaRS evolution

### 2. ProRS
- `10.1038/nchembio.790` - Halofuginone inhibits prolyl-tRNA synthetase
- `10.1016/j.cell.2004.09.030` - Noncanonical function of EPRS
- `10.1016/j.str.2015.02.011` - ProRS-Halofuginone structure

### 3. ThrRS
- `10.1016/s0092-8674(00)80746-1` - ThrRS-tRNA structure, zinc
- `10.1016/s0092-8674(00)00191-4` - tRNA-mediated editing in ThrRS
- `10.1038/75856` - Zinc-mediated amino acid discrimination

### 4. aaRS Family (Class I/II)
- `10.1038/347203a0` - Partition into two classes (Eriani 1990)
- `10.1126/science.2047877` - Class II AspRS structure
- `10.1038/347249a0` - SerRS structure (Class II)
- `10.1016/0092-8674(84)90278-2` - TyrRS double mutants
- `10.1038/s12276-018-0196-9` - TrpRS immune roles
- `10.1126/science.8128220` - SerRS-tRNA structure
- `10.1038/ng1727` - TyrRS CMT mutations
- `10.1093/emboj/cdf373` - Class I/II mode

### 5. Ancestral/LUCA
- `10.1007/s00239-015-9672-1` - Pre-LUCA aaRS ancestor
- `10.1073/pnas.92.7.2441` - Root of tree based on aaRS duplications
- `10.1038/nrg1324` - Resurrecting ancient genes
- `10.1038/nmicrobiol.2016.116` - LUCA physiology
- `10.1016/0022-2836(68)90392-6` - Origin of genetic code (Crick 1968)
- `10.1073/pnas.72.5.1909` - Co-evolution theory (Wong 1975)
- `10.1093/molbev/mst070` - Rodin-Ohno hypothesis evaluation

### 6. Disease
- `10.1038/nature05096` - Editing-defective aaRS causes neurodegeneration
- `10.1086/375039` - GlyRS mutations in CMT
- `10.1038/ng1727` - TyrRS CMT neuropathy
- `10.1073/pnas.0802862105` - aaRS connections to disease

### 7. Non-canonical Functions
- `10.1016/j.cell.2012.02.044` - LeuRS as leucine sensor (mTORC1)
- `10.1073/pnas.012602099` - aaRS regulates angiogenesis
- `10.1016/j.cell.2004.09.030` - Noncanonical EPRS function
- `10.1038/nchembio.937` - TrpRS-DNA-PKcs-PARP signaling

### 8. MSC (Multi-synthetase Complex)
- `10.1242/jcs.01342` - aaRS complexes beyond translation
- `10.1073/pnas.96.8.4488` - Genetic dissection of MSC
- `10.1074/jbc.rev118.002958` - MSC evolution and cancer

### 9. Promiscuity
- `10.1146/annurev-biochem-030409-143718` - Enzyme promiscuity review
- `10.1021/bi00660a026` - ValRS rejection of threonine (hyperspecificity)
- `10.1073/pnas.1019033108` - Misacylation of non-Met tRNAs
- `10.1093/nar/gkq763` - LeuRS editing pathways
- `10.1016/j.ymeth.2016.09.015` - Cognate vs non-cognate substrates

## Research Cluster Map

```
                    ┌─────────────────┐
                    │  aaRS EVOLUTION │
                    │   (Ancestral,   │
                    │    LUCA, Urzymes)│
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   STRUCTURE   │    │  YOUR FOCUS   │    │   FUNCTION    │
│  (Class I/II, │◄──►│ ProRS + ThrRS │◄──►│ (Non-canonical│
│   Family)     │    │               │    │   MSC)        │
└───────────────┘    └───────┬───────┘    └───────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  PROMISCUITY  │    │    DISEASE    │    │   MECHANISM   │
│ (Selectivity, │    │ (CMT, Cancer) │    │   (Editing)   │
│  Editing)     │    │               │    │               │
└───────────────┘    └───────────────┘    └───────────────┘
```

## Commands

### Check Progress
```bash
for dir in aars_fixed prorsrun thrrsrun aars_family ancestral_luca aars_disease aars_noncanonical aars_msc aars_promiscuity; do
  db="/tmp/$dir/candidates.db"
  [ -f "$db" ] && echo "$dir: $(sqlite3 $db 'SELECT COUNT(*) FROM paper_candidates')"
done
```

### Merge All Networks
```bash
python3 /tmp/merge_aars_networks.py
```

### Analyze Seed Importance
```bash
python3 /tmp/analyze_seed_importance.py
```

### View Merged Results
```bash
firefox /tmp/aars_merged/viewer.html
```

## Key Insight: Seed Selection

Seeds determine what literature you discover:
- **Urzyme seeds** → evolutionary/origin papers
- **ProRS/ThrRS seeds** → specific enzyme papers
- **Disease seeds** → clinical/mutation papers
- **Non-canonical seeds** → signaling/function papers

Different seeds = different networks = different perspectives on aaRS.
Merging gives you the COMPLETE picture of the field.

## Progress Snapshot (2026-01-17)

| Network | Papers | Status |
|---------|--------|--------|
| aaRS Urzymes | 32,209 | ✅ Complete |
| ProRS | ~10,000 | 🔄 Running |
| ThrRS | ~10,000 | 🔄 Running |
| Others | building | 🔄 Running |
| **TOTAL** | ~57,000+ | Growing |

## Next Steps

1. Wait for all jobs to complete (~20-30 min each)
2. Run merge script
3. Analyze seed importance
4. View unified aaRS network
5. Query the merged pool for specific topics
