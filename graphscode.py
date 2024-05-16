import matplotlib.pyplot as plt
import sys

fp = None
try:
    fp = open("10sgd/flres", "r")
except FileNotFoundError as err:
    print("File not Found"); sys.exit(1)
fldata = fp.readlines()
fp.close()
fp = None
try:
    fp = open("10sgd/sparres", "r")
except FileNotFoundError as err:
    print("File not Found"); sys.exit(1)
spdata = fp.readlines()
fp.close()
fp = None
try:
    fp = open("10sgd/tcres", "r")
except FileNotFoundError as err:
    print("File not Found"); sys.exit(1)
tcdata = fp.readlines()
fp.close()
flacc, fllos, flene, flsze = [], [], [], []
spacc, splos, spene, spsze = [], [], [], []
tcacc, tclos, tcene, tcsze = [], [], [], []
for flline in fldata:
    flline = flline.strip('\n')
    if flline.find("Accuracy") == 0: flacc.append(float(flline.split('(')[1].split(',')[0]))
    if flline.find("Loss") == 0: fllos.append(float(flline.split('(')[1].split(',')[0]))
    if flline.find("Energy") == 0: flene.append(float(flline.split('  ')[-1]))
    if flline.find("Bits") == 0: flsze.append(float(flline.split('  ')[-1]))
for spline in spdata:
    spline = spline.strip('\n')
    if spline.find("Accuracy") == 0: spacc.append(float(spline.split('(')[1].split(',')[0]))
    if spline.find("Loss") == 0: splos.append(float(spline.split('(')[1].split(',')[0]))
    if spline.find("Energy") == 0: spene.append(float(spline.split('  ')[-1]))
    if spline.find("Bits") == 0: spsze.append(float(spline.split('  ')[-1]))
for tcline in tcdata:
    tcline = tcline.strip('\n')
    if tcline.find("Accuracy") == 0: tcacc.append(float(tcline.split('(')[1].split(',')[0]))
    if tcline.find("Loss") == 0: tclos.append(float(tcline.split('(')[1].split(',')[0]))
    if tcline.find("Energy") == 0: tcene.append(float(tcline.split('  ')[-1]))
    if tcline.find("Bits") == 0: tcsze.append(float(tcline.split('  ')[-1]))

plt.figure(1, figsize=(20,20))
plt.title('Accuracy (10SGD)', fontsize=25)
plt.xlabel("Iterations", fontsize=16)
plt.ylabel("Accuracy (%)", fontsize=16)
plt.ylim(0, 100)
plt.plot(range(0, len(flacc)), flacc, alpha=0.8, label="base federated learning")
plt.plot(range(0, len(spacc)), spacc, alpha=0.8, label="FL with sparsification")
plt.plot(range(0, len(tcacc)), tcacc, alpha=0.8, label="FL with constant t-compression")
plt.grid(linestyle='dotted')
plt.locator_params(axis='y', nbins=10)
plt.legend(fontsize=16, loc="lower right")
plt.savefig("accuracy_iter", bbox_inches='tight')

plt.figure(2, figsize=(20,20))
plt.title('Loss (10SGD)', fontsize=25)
plt.xlabel("Iterations", fontsize=16)
plt.ylabel("Loss (%)", fontsize=16)
plt.ylim(0, 100)
plt.plot(range(0, len(fllos)), fllos, alpha=0.8, label="base federated learning")
plt.plot(range(0, len(splos)), splos, alpha=0.8, label="FL with sparsification")
plt.plot(range(0, len(tclos)), tclos, alpha=0.8, label="FL with constant t-compression")
plt.grid(linestyle='dotted')
plt.locator_params(axis='y', nbins=10)
plt.legend(fontsize=16, loc="lower right")
plt.savefig("los_iter", bbox_inches='tight')

plt.figure(3, figsize=(20,20))
plt.title('Cumulative Energy (10SGD)', fontsize=25)
plt.xlabel("Iterations", fontsize=16)
plt.ylabel("Energy (J)", fontsize=16)
plt.plot(range(0, len(flene)), flene, alpha=0.8, label="base federated learning")
plt.plot(range(0, len(spene)), spene, alpha=0.8, label="FL with sparsification")
plt.plot(range(0, len(tcene)), tcene, alpha=0.8, label="FL with constant t-compression")
plt.grid(linestyle='dotted')
plt.legend(fontsize=16)
plt.savefig("energy_iter", bbox_inches='tight')

plt.figure(4, figsize=(20,20))
plt.title('Gradient Size (10SGD)', fontsize=25)
plt.xlabel("Iterations", fontsize=16)
plt.ylabel("Size (bits)", fontsize=16)
plt.plot(range(0, len(flsze)), flsze, alpha=0.8, label="base federated learning")
plt.plot(range(0, len(spsze)), spsze, alpha=0.8, label="FL with sparsification")
plt.plot(range(0, len(tcsze)), tcsze, alpha=0.8, label="FL with constant t-compression")
plt.grid(linestyle='dotted')
plt.legend(fontsize=16)
plt.savefig("size_iter", bbox_inches='tight')

plt.figure(5, figsize=(20,20))
plt.title('Accuracy over Energy (10SGD)', fontsize=25)
plt.xlabel("Energy (J)", fontsize=16)
plt.ylabel("Accuracy (%)", fontsize=16)
plt.ylim(0, 100)
plt.plot(flene, flacc, alpha=0.8, label="base federated learning")
plt.plot(spene, spacc, alpha=0.8, label="FL with sparsification")
plt.plot(tcene, tcacc, alpha=0.8, label="FL with constant t-compression")
plt.grid(linestyle='dotted')
plt.locator_params(axis='y', nbins=10)
plt.legend(fontsize=16, loc="lower right")
plt.savefig("accurcy_energy", bbox_inches='tight')

plt.figure(6, figsize=(20,20))
plt.title('Loss over Energy (10SGD)', fontsize=25)
plt.xlabel("Energy (J)", fontsize=16)
plt.ylabel("Loss (%)", fontsize=16)
plt.ylim(0, 100)
plt.plot(flene, fllos, alpha=0.8, label="base federated learning")
plt.plot(spene, splos, alpha=0.8, label="FL with sparsification")
plt.plot(tcene, tclos, alpha=0.8, label="FL with constant t-compression")
plt.grid(linestyle='dotted')
plt.locator_params(axis='y', nbins=10)
plt.legend(fontsize=16)
plt.savefig("loss_energy", bbox_inches='tight')
