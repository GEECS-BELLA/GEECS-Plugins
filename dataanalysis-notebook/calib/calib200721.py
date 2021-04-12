#template
# 'camera name':{'sign':{'x':,'y':},'target':{'x':,'y':},'calib':,'unit':'mm'},

def calib():
    Dict={'scan':{},
          'Shotnumber':{},
          'DateTime Timestamp':{},
          'U_TempLegendOut Temperature':{},
          'U_TempStretcher Temperature':{},
          'UC_ModeImager centroid':{'target':{'x':580,'y':419},'calib':0.00255,'unit':'mm'},
          'UC_GhostNear centroid':{'calib':0.006217,'unit':'mm'},
          'UC_GhostFar centroid':{'calib':0.006217, 'unit':'mm'},
          #pulsed beam
          'UC_StretcherOut centroid':{'sign':{'x':-1,'y':-1},'calib':0.0138,'unit':'mm'},
          'UC_Amp2_IR_input centroid':{'sign':{'x':-1,'y':-1},'target':{'x':215,'y':247},'calib':0.0261,'unit':'mm'},
          'UC_Amp3_IR_input centroid':{'sign':{'x':1,'y':-1},'target':{'x':341,'y':421},'calib':0.0293,'unit':'mm'},
          'UC_Amp4_IR_input centroid':{'sign':{'x':-1,'y':1},'target':{'x':298,'y':290},'calib':0.071,'unit':'mm'},
          'UC_ExpanderIn1_Pulsed centroid':{'sign':{'x':-1,'y':1},'target':{'x':201,'y':251},'calib':0.150,'unit':'mm'},
          'UC_GratingMode centroid':{'sign':{'x':1,'y':1},'target':{'x':155,'y':158},'calib':0.290,'unit':'mm'},
          'UC_TC_Phosphor centroid':{'sign':{'x':1,'y':1},'target':{'x':752,'y':413},'calib':0.12,'unit':'mrad'},
          #'UC_TC_Phosphor centroid':{'sign':{'x':1,'y':1},'target':{'x':752,'y':413},'calib':0.037,'unit':'mm'},
          
          #diode beam
          'UC_ExpanderIn2 centroid':{'sign':{'x':1,'y':1},'target':{'x':206,'y':214},'calib':0.0348,'unit':'mm'},
          'UC_TubeIn centroid':{'sign':{'x':1,'y':1},'target':{'x':51,'y':49},'calib':0.148,'unit':'mm'},
          'UC_CompIn centroid':{'sign':{'x':1,'y':1},'target':{'x':203,'y':200},'calib':0.057,'unit':'mm'},
          'UC_BCaveIn centroid':{'sign':{'x':1,'y':1},'target':{'x':171,'y':140},'calib':0.2,'unit':'mm'},
          'UC_OAPin1 centroid':{'sign':{'x':1,'y':1},'target':{'x':183,'y':215},'calib':0.194,'unit':'mm'},
          'UC_OAPin2 centroid':{'sign':{'x':-1,'y':1},'target':{'x':459, 'y':462},'calib':{'x':0.1516,'y':0.193},'unit':'mm'},
          'UC_Beamline_Out centroid':{'sign':{'x':1,'y':1},'target':{'x':214,'y':202},'calib':0.154,'unit':'mm'}}
    return Dict



#OAPin2 #y: 0.193