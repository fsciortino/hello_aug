'''Script to demonstrate how to make a scope/cview/reviewplus-like set of plots showing time traces for an AUG discharge. 

These are just examples. You can substitute any fields (or add more) by selecting shotfiles from the list found on the ISIS website:
https://www.aug.ipp.mpg.de/cgibin/sfread_only/isis?action=ListOfKnownDiags

The data available in each shotfile can be browsed online from
https://www.aug.ipp.mpg.de/cgibin/sfread_only/isis
'''
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import aug_sfutils as sf
import sys, os

try:
    shot = int(sys.argv[1])
except:
    shot = 38996

data = {}
# some general detachment metrics
mac = sf.SFREAD(shot, 'MAC')
if mac.status:
    data['mac'] = {}
    data['mac']['time_Tdiv'] = mac.gettimebase('Tdiv')
    data['mac']['Tdiv'] = mac.getobject('Tdiv')
    data['mac']['time_Ipolsol'] = mac.gettimebase('Ipolsola')
    data['mac']['Ipolsola'] = mac.getobject('Ipolsola')
    data['mac']['Ipolsoli'] = mac.getobject('Ipolsoli')

# johann
jow = sf.SFREAD(shot, 'JOW')
if jow.status:
  data['jow'] = {}
  data['jow']['time'] = jow.gettimebase('c_W_44')
  data['jow']['c_W_44'] = jow.getobject('c_W_44')
  data['jow']['c_W_45'] = jow.getobject('c_W_45')
  data['jow']['c_W_46'] = jow.getobject('c_W_46')

# SPRED
scl = sf.SFREAD(shot, 'SCL')
if scl.status:
    data['scl'] = {}
    data['scl']['time'] = scl.gettimebase('W44_133')
    data['scl']['W_44'] = scl.getobject('W44_133')

# Gracing incidence
giw = sf.SFREAD(shot, 'GIW')
if giw.status:
    data['giw'] = {}
    data['giw']['time'] = giw.gettimebase('c_W')
    data['giw']['c_W'] = giw.getobject('c_W')
    data['giw']['c_W_l'] = giw.getobject('c_W_l')

# density
dcn = sf.SFREAD(shot, 'DCN')
if dcn.status:
    data['dcn'] = {'time_H1': dcn.gettimebase('H-1'),
                   'H-1': dcn.getobject('H-1'),
                   'time_H5': dcn.gettimebase('H-5'),
                   'H-5': dcn.getobject('H-5')
    }
    
# overall radiation
bpd = sf.SFREAD(shot, 'BPD')
if bpd.status:
    data['bpd'] = {}
    data['bpd']['time'] = bpd.gettimebase('Prad')
    data['bpd']['Prad'] = bpd.getobject('Prad')

# Zeff
idz = sf.SFREAD(shot, 'IDZ')
if idz.status:
    data['idz'] = {}
    data['idz']['time'] = idz.gettimebase('Zeff')
    data['idz']['Zeff'] = idz.getobject('Zeff')

# heating systems
ecs = sf.SFREAD(shot, 'ECS')
if ecs.status:
    data['ecs'] = {}
    data['ecs']['time'] = ecs.gettimebase('PECRH')
    data['ecs']['Pecrh'] = ecs.getobject('PECRH')
    
icp = sf.SFREAD(shot, 'ICP')
if icp.status:
    data['icp'] = {}
    data['icp']['time'] = icp.gettimebase('PICRN')
    data['icp']['Picrh'] = icp.getobject('PICRN')
    
nis = sf.SFREAD(shot, 'NIS')
if nis.status:
    data['nis'] = {}
    data['nis']['time'] = nis.gettimebase('PNI')
    data['nis']['Pnbi'] = nis.getobject('PNI')

# ----------------------------
# now plot
fig,axs = plt.subplots(4,1, figsize=(8,8), sharex=True)
fig.suptitle(fr'shot #{shot}')
if 'mac' in data:
    axs[0].plot(data['mac']['time_Ipolsol'], data['mac']['Ipolsola'], label='Ipolsola')
    axs[0].plot(data['mac']['time_Ipolsol'], data['mac']['Ipolsoli'], label='Ipolsoli')

if 'giw' in data:
    axs[1].semilogy(data['giw']['time'], data['giw']['c_W'], label='c_W')
    axs[1].semilogy(data['giw']['time'], data['giw']['c_W_l'], label='c_W_l')
if 'jow' in data:
    axs[1].semilogy(data['jow']['time'], data['jow']['c_W_44'], label='c_W44+ JOW')
    axs[1].semilogy(data['jow']['time'], data['jow']['c_W_45'], label='c_W45+ JOW')
    axs[1].semilogy(data['jow']['time'], data['jow']['c_W_46'], label='c_W46+ JOW')
if 'scl' in data:
    axs[1].semilogy(data['scl']['time'], np.nanmax(data['giw']['c_W'])*data['scl']['W_44']/np.nanmax(data['scl']['W_44']), label='c_W44+ (proxy)')

    

if 'dcn' in data:
    axs[2].plot(data['dcn']['time_H1'], data['dcn']['H-1']/1e19, label='H-1', c='r')
    axs[2].plot(data['dcn']['time_H5'], data['dcn']['H-5']/1e19, label='H-5', c='b')
    axs[2].set_ylabel(r'$\int n_e dl$ [$1^{19}$ $m^{-2}$]')
if 'idz' in data:
    ax2 = axs[2].twinx()
    color = 'tab:green'
    ax2.set_ylabel('Zeff', color=color)
    # no variation saved over radius; radial average
    ax2.plot(data['idz']['time'], np.mean(data['idz']['Zeff'],axis=1), label='Zeff', c=color)
    ax2.tick_params(axis='y', labelcolor=color)
    

if 'ecs' in data:
    axs[3].plot(data['ecs']['time'], data['ecs']['Pecrh']/1e6, label=r'$P_{ECRH}$ [MW]')
if 'icp' in data:
    axs[3].plot(data['icp']['time'], data['icp']['Picrh']/1e6, label=r'$P_{ICRH}$ [MW]')
if 'nis' in data:
    axs[3].plot(data['nis']['time'], data['nis']['Pnbi']/1e6, label=r'$P_{NBI}$ [MW]')
if 'bpd' in data:
    axs[3].plot(data['bpd']['time'], data['bpd']['Prad']/1e6, label=r'$P_{rad}$ [MW]')


for ax in axs.flatten():
    ax.legend(loc='best').set_draggable(True)
axs[3].set_xlabel('t [s]')
axs[3].set_xlim([0,6])
#axs[3].set_ylim([0,1])
plt.tight_layout()
