import numpy as np
import matplotlib.pyplot as plt
''' VIRS BURGERS
a=[np.log10(6.28e-05),np.log10(0.0002),np.log10(0.00036),np.log10(0.00069),np.log10(0.0033)]
b=[np.log10(28979),np.log10(10996),np.log10(7072),np.log10(3923),np.log10(1160)]
a1=[np.log10(3.96e-06),np.log10(4.49e-06),np.log10(8.85e-06),np.log10(2.5e-05),np.log10(0.00053)]
b1=[np.log10(28980),np.log10(10997),np.log10(7066),np.log10(3796),np.log10(1202)]
a2=[np.log10(4.7e-05),np.log10(0.00015),np.log10(0.00027),np.log10(0.00054),np.log10(0.0026)]
b2=[np.log10(27979),np.log10(10330),np.log10(6779),np.log10(3779),np.log10(1103)]
plt.plot(a1,b1,'bs--',alpha=0.8,linewidth=2,label='NTRKC V1')
plt.plot(a,b,'r*--',alpha=0.8,linewidth=2,label='two step RKC')
plt.plot(a2,b2,'y^--',alpha=0.8,linewidth=2,label='RKC2')
plt.legend()
plt.xlabel("$log_{10}(err)$")
plt.ylabel("$log_{10}(Nfe)$")
#plt.grid()
#plt.savefig('C:/Users/A204-7/Desktop/论文撰写及其模板模板/numberical_results/visburgers.eps')
plt.legend()
plt.xlabel("$log_{10}(err)$")
plt.ylabel("$log_{10}(Nfe)$")
'''

'''
#FHN
a=[2.4e-05,1.07e-05,5.04e-06,1.8e-06,1.2e-07,2.63e-08,5.68e-09,8.9e-10,2.2e-10,2.45e-11]
a1=[1408,1744,2332,3340,7008,10004,14008,23332,35008,70008]
b=[1.4e-05,5.7e-06,5.6e-06,5.5e-06,2.76e-06,1.6e-06,1.3e-06,5.5e-07,5.5e-07,2.75e-07]
b1=[1306,1618,2164,3100,6506,9288,13006,21664,32506,65006]
c=[1.52e-06,5.7e-07,1.73e-07,3.95e-08,5.06e-09,2.68e-09,2.12e-09,4.45e-10,8.4e-10,4.19e-10]

d=[1.72e-06,4.84e-07,2.33e-07,2.71e-08,1.88e-08]
d1=[3932,4709,6081,7355,8825]
e=[5.60e-06,4.46e-06,3.26e-06,2.554e-06,1.59e-06]
e1=[3932,4709,6081,7355,8825]
plt.plot(np.log10(d),np.log10(d1),'bs--',alpha=0.8,linewidth=2,label='NTRKC V1')
#plt.plot(np.log10(e),np.log10(d1),'go--',alpha=0.8,linewidth=2,label='NTRKC V2')
plt.plot(np.log10(e),np.log10(e1),'r*--',alpha=0.8,linewidth=2,label='two step RKC')
plt.legend()
plt.xlabel("$log_{10}(err)$")
plt.ylabel("$log_{10}(Nfe)$")
#plt.savefig('C:/Users/A204-7/Desktop/论文撰写及其模板模板/numberical_results/FHN.eps')
plt.show()
'''


''' ade
a=[9.7e-06,1.3e-06,5.75e-07,1.4e-07,1.7e-08]
a1=[4358,7459,8834,13159,28959]
b=[0.003,7.6e-06,1.55e-06,7.9e-07,2.2e-08]
b1=[4358,7459,8834,13159,28959]
c=[0.74,0.057,0.014,0.00048,3.75e-08]
c1=[4009,6959,8334,12359,27459]
plt.plot(np.log10(a),np.log10(a1),'bs--',alpha=0.8,linewidth=2,label='NTRKC V1')
plt.plot(np.log10(c),np.log10(c1),'y^--',alpha=0.8,linewidth=2,label='RKC2')
plt.plot(np.log10(b),np.log10(b1),'r*--',alpha=0.8,linewidth=2,label='two step RKC')
plt.legend()
plt.xlabel("$log_{10}(err)$")
plt.ylabel("$log_{10}(Nfe)$")
#plt.savefig('C:/Users/A204-7/Desktop/论文撰写及其模板模板/numberical_results/adebt1gamma4000.eps')
plt.show()'''
'''
a=[0.0017,1.0e-05,8.6e-07,9.89e-08,2.8e-09,5e-10]
a1=[2119,4209,7159,12759,27959,52958]
b=[0.0004,5.2e-05,2.1e-06,5.3e-07,7.6e-09,7.5e-10]
b1=[2119,4209,7159,12759,27959,52958]
c=[4.45e-05,6.13e-06,1.19e-06,6.46e-07,1.1e-08,1.7e-9]
c1=[1939,3859,6759,12159,27459,50959]
plt.plot(np.log10(a),np.log10(a1),'bs--',alpha=0.8,linewidth=2,label='NTRKC V1')
plt.plot(np.log10(c),np.log10(c1),'y^--',alpha=0.8,linewidth=2,label='RKC2')
plt.plot(np.log10(b),np.log10(b1),'r*--',alpha=0.8,linewidth=2,label='two step RKC')
plt.legend()
plt.xlabel("$log_{10}(err)$")
plt.ylabel("$log_{10}(Nfe)$")
#plt.savefig('C:/Users/A204-7/Desktop/论文撰写及其模板模板/numberical_results/adebt0gamma1000.eps')
plt.show()'''
'''
a=[1.81e-05,4.2e-05,3.06e-06,9.3e-06,3.38e-06]
a1=[774,832,889,978,1075]
b=[0.0011,0.0055,0.0002,0.003,0.0001]
b1=[774,832,889,978,1075]
c=[0.0047,0.0033,0.0019,0.00099,0.0018]
c1=[2076,2530,3430,5896,6490]
plt.figure(figsize=(10,8))
plt.plot(np.log2(a),np.log2(a1),'bs--',alpha=0.8,linewidth=2,label='NTRKC')
#plt.plot(np.log10(b),np.log10(1),'y^--',alpha=0.8,linewidth=2,label='two step RKC')
plt.plot(np.log2(b),np.log2(b1),'r*--',alpha=0.8,linewidth=2,label='two step RKC')
plt.legend(loc=1)
plt.xlabel("$log_{10}(err)$")
plt.ylabel("$log_{10}(Nfe)$")

#plt.savefig('C:/Users/A204-7/Desktop/论文撰写及其模板模板/numberical_results/burssrd.eps')
plt.show()
'''
'''
#ade
a=[9.59e-06,1.35e-06,5.75e-07,1.49e-07,7.22e-8,1.75e-8]
a1=[4359,7459,8834,13159,15708,28459]
b=[0.003,7.7e-06,1.55e-06,7.49e-07,3.32e-07,2.24e-08]
b1=[4359,7459,8834,13159,15708,28459]
c=[7.2e-06,9.7e-07,4.9e-07,1.1e-07,5.69e-08,1.75e-08]
c1=[4359,7459,8834,13159,15708,28459]
d=[6.74e-06,7.86e-07,4.16e-07,9.06e-08,4.69e-08,1.75e-08]
d1=[4509,7659,9087,13359,16204,28959]
plt.plot(np.log10(a),np.log10(a1),'bs--',alpha=0.8,linewidth=2,label='NTRKC')
plt.plot(np.log10(b),np.log10(b1),'r*--',alpha=0.8,linewidth=2,label='two step RKC')
#plt.plot(np.log10(c),np.log10(a1),'ys--',alpha=0.8,linewidth=2,label='NTRKCv1')
plt.plot(np.log10(d),np.log10(d1),'ys--',alpha=0.8,linewidth=2,label='NTRKCv2')
plt.legend()
plt.xlabel("$log_{10}(err)$")
plt.ylabel("$log_{10}(Nfe)$")
#plt.savefig('C:/Users/A204-7/Desktop/论文撰写/numberical_results/adegama4000v1.eps')
plt.show()
'''

#burss
#a=[0.00011391,0.00029748,8.42934268e-05,1.39929774e-05,6.28291557e-05,1.94188406e-05,0.00017834]
#a1=[4458, 5397,6330,7407, 8280,  9192,10198]
b=[0.0002,0.00035,0.00026,0.00022,0.00027,2.76e-05,6.93e-05,9.9e-05]
b1=[13859,16107,18824,26647,39157,63923,76642,89430]
c=[0.00032,4.92e-05,3.47e-05,1.16e-05,9.79e-06,7.06e-06,5.41e-06,4.04e-06,3.24e-06,2.27e-06,1.65e-06]
c1=[5459,10160,15005,19606,37606,42306,46960,51706,68960,78160,87360]

#d=[6.74e-06,7.86e-07,4.16e-07,9.06e-08,4.69e-08,1.75e-08]
#d1=[4509,7659,9087,13359,16204,28959]
#plt.plot(a,(a1),'bs--',alpha=0.8,linewidth=2,label='NTRKC')
plt.plot(np.log10(b),np.log10(b1),'r*--',alpha=0.8,linewidth=2,label='two step RKC')
#plt.plot(np.log10(c),np.log10(a1),'ys--',alpha=0.8,linewidth=2,label='NTRKCv1')
plt.plot(np.log10(c),np.log10(c1),'bs--',alpha=0.8,linewidth=2,label='NTRKCv2')
plt.legend()
plt.xlabel("$log_{10}(err)$")
plt.ylabel("$log_{10}(Nfe)$")
#plt.savefig('C:/Users/A204-7/Desktop/论文撰写/numberical_results/adegama4000v1.eps')
plt.show()

'''
#ROBER
a=[6.0e-07,1.66e-07,7.69e-08,2.82e-08,1.52e-08,7.41e-09,3.02e-9,9.37e-10,1.62e-10]
a1=[42841,72699,87885,112637,131108,156805,195059,257975,379731]#380815]
b=[8.87e-07,6.18e-07,3.88e-07,3.77e-07,3.44e-07,3.2e-07,2.22e-07,1.44e-07]#8.88e-08]
b1=[66664,87853,112637,131104,195054,257975,380824,420899]
c=[7.37e-07,1.0e-08,9.31e-09,4.01e-09,1.72e-09,9.2e-10,2.35e-11,1.08e-11]
c1=[42841,87886,97531,112637,132622,156806,199921,399855]
#d=[6.74e-06,7.86e-07,4.16e-07,9.06e-08,4.69e-08,1.75e-08]
#d1=[4509,7659,9087,13359,16204,28959]
plt.plot(np.log10(a),np.log10(a1),'bs--',alpha=0.8,linewidth=2,label='NTRKC')
plt.plot(np.log10(c),np.log10(c1),'g^--',alpha=0.8,linewidth=2,label='NTRKCⅠ')
plt.plot(np.log10(b),np.log10(b1),'r*--',alpha=0.8,linewidth=2,label='two step RKC')
plt.legend()
plt.xlabel("$log_{10}(err)$")
plt.ylabel("$log_{10}(Nfe)$")
#plt.savefig('C:/Users/A204-7/Desktop/论文撰写/numberical_results/ROBER.eps')
plt.show()
'''

'''
#nostiff
a=[10  ,     
 20 ,     
 40  ,   
 80  ,  
 160  , 
 320  ,
640,
 1280,
 2560]
a1=[3.16282530e-02,
  3.85348054e-03,
 3.53570819e-04,
 3.04588124e-05,
 2.71084023e-06,
 2.62169185e-07,
 2.77079452e-08,
 3.13006582e-09,
 3.67465613e-10]
a2=[3.03697986,
 3.44609099,
 3.53706762,
 3.4900477 ,
 3.37017006,
 3.24212649,
 3.14603482,
 3.09051184]
b=[1.69809120e-02,
 3.58720461e-03,
 8.15124817e-04,
 1.94440827e-04,
 4.75210466e-05,
 1.17495668e-05,
 2.92140855e-06,
 7.28376697e-07,
 1.81848644e-07]#8.88e-08]
b1=[10  ,     
 20 ,     
 40  ,   
 80  ,  
 160  , 
 320  ,
640,
 1280,
 2560]
b2=[2.242982  ,
 2.13776714,
 2.06768981,
 2.03269266,
 2.01595904,
 2.00787153,
 2.00390746,
 2.0019466 ]
c=[1.13048167e-02,
 1.41476533e-03,
 1.69046825e-04,
 2.03612656e-05,
 2.48872740e-06,
 3.07181585e-07,
 3.81020281e-08,
 4.73261219e-09,
 5.86770410e-10]
c1=[10  ,     
 20 ,     
 40  ,   
 80  ,  
 160  , 
 320  ,
640,
 1280,
 2560]
c2=[2.99830293,
 3.06506794,
 3.05352378,
 3.03234712,
 3.01824457,
 3.01115204,
 3.00915918,
 3.01176868]
#d=[6.74e-06,7.86e-07,4.16e-07,9.06e-08,4.69e-08,1.75e-08]
#d1=[4509,7659,9087,13359,16204,28959]
plt.plot(np.log10(a1),np.log2(a),'bs--',alpha=0.8,linewidth=2,label='NTRKC')
plt.plot(np.log10(b),np.log2(b1),'y^--',alpha=0.8,linewidth=2,label='RKC2')
plt.plot(np.log10(c),np.log2(c1),'r*--',alpha=0.8,linewidth=2,label='two step RKC')
plt.legend()
plt.xlabel("$log10(err)$")
plt.ylabel("$log2(steps)$")
#plt.savefig('C:/Users/A204-7/Desktop/论文撰写/numberical_results/nostifferr.eps')
plt.show()'''