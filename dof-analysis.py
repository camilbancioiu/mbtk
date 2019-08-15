from pathlib import Path
import pickle

citrfolder = Path('testfiles', 'tmp', 'test_ipcmb_with_gtests', 'ci_test_results')
template = 'alarm_{}_{}_dof_{}.pickle'

target = 19

funoptimized = citrfolder / template.format(target, 'unoptimized', 'cjpi')
fadtree = citrfolder / template.format(target, 'adtree', 'cjpi')
fdcmi = citrfolder / template.format(target, 'dcmi', 'cjpi')

with funoptimized.open('rb') as f:
    citrs_unoptimized = pickle.load(f)

with fadtree.open('rb') as f:
    citrs_adtree = pickle.load(f)

with fdcmi.open('rb') as f:
    citrs_dcmi = pickle.load(f)

print(len(citrs_unoptimized))
print(len(citrs_adtree))
print(len(citrs_dcmi))

c = 1000

for (un, ad, dcmi) in zip(citrs_unoptimized, citrs_adtree, citrs_dcmi):
    un_ad = (un == ad)
    un_dcmi = (un == dcmi)
    ad_dcmi = (ad == dcmi)
    if not (un_ad and un_dcmi and ad_dcmi):
        print(un)
        print(ad)
        print(dcmi)
        print()
    c -= 1
    if c == 0:
        break

accuracy_un = (len([r for r in citrs_unoptimized if r.accurate()]) / len(citrs_unoptimized))
accuracy_adtree = (len([r for r in citrs_adtree if r.accurate()]) / len(citrs_adtree))
accuracy_dcmi = (len([r for r in citrs_dcmi if r.accurate()]) / len(citrs_dcmi))

print('accuracy_un:', accuracy_un)
print('accuracy_adtree:', accuracy_adtree)
print('accuracy_dcmi:', accuracy_dcmi)
