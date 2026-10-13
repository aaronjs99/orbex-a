# /***********************************************************
# *                                                         *
# * Copyright (c) 2022                                      *
# *                                                         *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu, Brett T. Lopez                *
# * Contact: {aaronjs, btlopez}@ucla.edu                    *
# *                                                         *
# ***********************************************************/

# PACKAGE IMPORTS
import numpy             as np
import matplotlib.pyplot as plt
from   scipy      import optimize  as opt
from   params     import *
from   commons    import filenameCreator
from   orbitsim   import simulate
from   taskalloc  import genNeighbors, GreedyAgent
from   spacecraft import Spacecraft, Target, Chaser
from   deflection import targetDeflect
np.random.seed(0)

# MAIN PROGRAM
if __name__ == '__main__': 
  from params import *
  
  def cylinderRadialConstraint(x, r, f):
    return abs(np.sum([r[j]**2 for j in range(0, 2)]) - targetLimit['r_T']**2) -                0.1**2

  def cylinderAxialConstraint(x, r, f):
    return abs(np.sum([r[j]**2 for j in range(2, 3)]) -                  0**2) - targetLimit['l_T']**2

  def ellipsoidConstraint(x, r, f):
    ellRad = [targetLimit['r_Tx'], targetLimit['r_Ty'], targetLimit['r_Tz']]
    return abs(np.sum([r[j]**2/ellRad[j]**2 for j in range(0, 3)]) -  1**2) -  0.01

  numTimes   = int(totalTime/Spacecraft.dt)
  T = Target({"name": 'Target', "numStates": 3, "initState": th_T0},
             {"angularVelocity": w_T0, "momInertia": I_T0})
  
  if   targetShape == 'cylinder':
    shapeParams = {'cylHeight': targetLimit['l_T'],
                   'cylRadius': targetLimit['r_T'],
                   'cylCenter': targetCenter}
    T.updateGeometry(geometryIneqs = [cylinderRadialConstraint, cylinderAxialConstraint])
  elif targetShape == 'ellipsoid':
    shapeParams = {'ellRadX'  : targetLimit['r_Tx'],
                   'ellRadY'  : targetLimit['r_Ty'],
                   'ellRadZ'  : targetLimit['r_Tz'],
                   'ellCenter': targetCenter}
    T.updateGeometry(geometryIneqs = [ellipsoidConstraint])
  T.updateState(numSteps = initialTimeLapse)
  
  C = []
  for i in range(numChasers):
    C.append(Chaser({"name": 'Chaser '+str(i+1),
                     "numStates": 6,
                     "initState": np.append([np.random.uniform(-1.0e+3, 1.0e+3) for i in range(3)],
                                            [np.random.uniform(-1.0e+1, 1.0e+1) for i in range(3)])}))
  
  ### Target Motion Test
  TStopSteps = int(TStopTime/Spacecraft.dt)
  target_test = True
  if target_test:
    T, _, _, _ = targetDeflect(target        = T,
                               dt            = dt,
                               x_f           = np.array([0, 0, 0, 0, 0, 0]),
                               bounds        = (forceBounds),
                               numSteps      = TStopSteps,
                               numChasers    = numChasers,
                               rLen          = 3,
                               fLen          = 3,
                               chaserMinDist = chaserMinDist,
                               shapeParams   = shapeParams,)
    exit()
  
  ### Execution Loop
  if debug_section:
    print('EXECUTION LOOP')
  for t in np.arange(0, totalTime, iterTime):
    if debug_section:
      print('  Time = ', t)
    ### Target updates its state
    T.updateState(numSteps = numUpdateSteps)

    ### DETECTION
    detectionPhase = True
    if detectionPhase:
      if debug_section:
        print('  DETECTION')
      ### Chasers reset information from previous cycle
      if debug_section:
        print('    Chaser Information Reset')
      for chaser in C:
        chaser.resetInfo()
      
      ### Chasers update their states
      if debug_section:
        print('    Chaser State Update')
      for chaser in C:
        chaser.updateState(numUpdateSteps)
      
      ### Chasers update their neighbor lists
      if debug_section:
        print('    Chaser Neighbor List Update')
      G = genNeighbors('maxDist', [chaser.currState for chaser in C], neighborMaxDist, idOffset = True)
      for chaser in C:
        chaser.updateNeighbors('set', G[chaser.id-1])
      
      ### Chasers update their lists of neighbor locations
      if debug_section:
        print('    Chaser Neighbor Location Update')
      R_0G = {}
      for chaser in C:
        R_0G[chaser.id] = {chaser.id: chaser.currState[:3], 'len': 1}
      while not all([chaser.neighborsLocnConsensus for chaser in C]):
        for chaser in C:
          for neighbor in chaser.neighbors:
            chaser.commData('get_locations', R_0G[neighbor])
          R_0G[chaser.id] = chaser.rNInfo
      
      ### Chasers perform target state estimation using consensus
      debug_in_loop = False
      if debug_section:
        print('    Target State Estimation')
      tAngPosList = {}
      tAngVelList = {}
      for chaser in C:
        tAngPosList[chaser.id] = []
        tAngVelList[chaser.id] = []
      while True:
        ### Chasers observe the Target
        if debug_section and debug_in_loop:
          print('      Chaser Target Observation')
        tInfo = {}
        for chaser in C:
          tInfo[chaser.cid] = chaser.observeTarget(T)
          tAngPosList[chaser.cid].append(tInfo[chaser.cid][0])
          tAngVelList[chaser.cid].append(tInfo[chaser.cid][1])
    
        ### Chasers communicate target info with neighbors
        if debug_section and debug_in_loop:
          print('      Chaser Target Info Communication')
        for chaser in C:
          for neighbor in chaser.neighbors:
            chaser.commData('target', tInfo[neighbor], neighbor)

        ### Chasers come to consensus about target state
        if debug_section and debug_in_loop:
          print('      Chaser Target Info Consensus')
        if all([chaser.targetObserveConsensus for chaser in C]):
          break
      if debug_plots:
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title('Target Angular Position')
        for chaser in C:
          ax.plot3D([x[0] for x in tAngPosList[chaser.cid]],
                    [x[1] for x in tAngPosList[chaser.cid]],
                    [x[2] for x in tAngPosList[chaser.cid]],
                    label = 'Chaser '+str(chaser.cid))
        ax.scatter(T.currState[0],
                   T.currState[1],
                   T.currState[2],
                   label = 'Target')
        plt.show()
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title('Target Angular Velocity')
        for chaser in C:
          ax.plot3D([x[0] for x in tAngVelList[chaser.cid]],
                    [x[1] for x in tAngVelList[chaser.cid]],
                    [x[2] for x in tAngVelList[chaser.cid]],
                    label = 'Chaser '+str(chaser.cid))
        ax.scatter(T.angularVelocity[0],
                   T.angularVelocity[1],
                   T.angularVelocity[2],
                   label = 'Target')
        plt.show()
        plt.figure()
        plt.title('Target Angular Position')
        plt.subplot(3,1,1)
        plt.ylabel('X (rad)')
        plt.xlabel('Time (s)')
        for chaser in C:
          plt.plot(np.arange(0, len(tAngPosList[chaser.cid])*Spacecraft.dt, Spacecraft.dt),
                   [x[0] for x in tAngPosList[chaser.cid]],
                   label = 'Chaser '+str(chaser.cid))
        plt.subplot(3,1,2)
        plt.ylabel('Y (rad)')
        plt.xlabel('Time (s)')
        for chaser in C:
          plt.plot(np.arange(0, len(tAngPosList[chaser.cid])*Spacecraft.dt, Spacecraft.dt),
                   [x[1] for x in tAngPosList[chaser.cid]],
                   label = 'Chaser '+str(chaser.cid))
        plt.subplot(3,1,3)
        plt.ylabel('Z (rad)')
        plt.xlabel('Time (s)')
        for chaser in C:
          plt.plot(np.arange(0, len(tAngPosList[chaser.cid])*Spacecraft.dt, Spacecraft.dt),
                   [x[2] for x in tAngPosList[chaser.cid]],
                   label = 'Chaser '+str(chaser.cid))
        plt.show()
        plt.figure()
        plt.title('Target Angular Velocity')
        plt.subplot(3,1,1)
        plt.ylabel('X (rad/s)')
        plt.xlabel('Time (s)')
        for chaser in C:
          plt.plot(np.arange(0, len(tAngVelList[chaser.cid])*Spacecraft.dt, Spacecraft.dt),
                   [x[0] for x in tAngVelList[chaser.cid]],
                   label = 'Chaser '+str(chaser.cid))
        plt.subplot(3,1,2)
        plt.ylabel('Y (rad/s)')
        plt.xlabel('Time (s)')
        for chaser in C:
          plt.plot(np.arange(0, len(tAngVelList[chaser.cid])*Spacecraft.dt, Spacecraft.dt),
                   [x[1] for x in tAngVelList[chaser.cid]],
                   label = 'Chaser '+str(chaser.cid))
        plt.subplot(3,1,3)
        plt.ylabel('Z (rad/s)')
        plt.xlabel('Time (s)')
        for chaser in C:
          plt.plot(np.arange(0, len(tAngVelList[chaser.cid])*Spacecraft.dt, Spacecraft.dt),
                   [x[2] for x in tAngVelList[chaser.cid]],
                   label = 'Chaser '+str(chaser.cid))
        plt.show()
    detectionPhase = False
    
    ### DECOUPLED MODE
    if decoupledMode:
      if debug_section:
        print('  DECOUPLED MODE')
      ### DECISION
      decisionPhase = True
      if decisionPhase:
        if debug_section:
          print('  DECISION')
        ### Chasers calculate goal locations via single-agent optimization of individually-run system simulations and multi-agent communication
        if debug_section:
          print('    Goal Location Calculation')
        while True:
          ### Chasers propose goal locations
          if debug_section:
            print('      Goal Location Proposal')
          if not any([chaser.goalCalculateConsensus for chaser in C]):
            GInfo = {}
            for chaser in C:
              GInfo[chaser.cid] = chaser.calculateGoals()
          
          ### Chasers communicate goal info with neighbors
          if debug_section:
            print('      Goal Location Communication')
          for chaser in C:
            for neighbor in chaser.neighbors:
              GInfo[neighbor]   =        chaser.commData('goal_list', GInfo[neighbor], neighbor)
              GInfo[chaser.cid] = C[neighbor-1].commData('goal_list', GInfo[neighbor], chaser.cid)
          
          ### Chasers come to consensus about goal locations
          if debug_section:
            print('      Goal Location Consensus')
          if all([chaser.goalCalculateConsensus for chaser in C]):
            gInfo = list(GInfo[1]['goalLocations'])
            gInfo = [gInfo[i:i+3] for i in range(0, len(gInfo), 3)]
            for chaser in C:
              chaser.goalCalculateConsensus = False
            break
        
        ### Chasers perform distributed task allocation to determine which chaser will go to which goal
        if debug_section:
          print('    Task Allocation')
        R_T    = []
        agents = []
        for chaser in C:
          R_T   .append(chaser.determineGoalInit('pick_prelim_goal'))
        for chaser in C:
          agents.append(chaser.determineGoalInit('create_agent'    , GreedyAgent))
        complete = False
        while not complete:
          for chaser in C:
            w = chaser.id - 1
            agents, agent_w, _ = agents[w].greedySingleLoop(agents)
            agents[w] = agent_w
          complete = not(any([agent.bool_int for agent in agents]))
        for chaser in C:
          chaser.goalState = chaser.taskAllocAgent.r_T
          chaser.goalState = np.append(chaser.goalState, [0, 0, 0])
      decisionPhase = False
      
      ### EXECUTION
      executionPhase = True
      if executionPhase:
        if debug_section:
          print('  EXECUTION')
        ### Chasers calculate input 
        if debug_section:
          print('    Input Calculation')
        for chaser in C:
          chaser.determineInputs(numUpdateSteps)
      executionPhase = False
    
    ### SYSTEM REQUIREMENT MODIFICATION
    if debug_section:
      print('  SYSTEM REQUIREMENT MODIFICATION')
    for chaser in C:
      if debug_section:
        print('    Goal Bound Recalculation')
      chaser.goalBounds = (chaser.goalBounds[0]/2, chaser.goalBounds[1]/2)
  
  ### Plotting
  sep_all_plots = False
  orbit_sim     = False
  if sep_all_plots:
    T.plotStateHistory(params = {'sep_plots': False, 'disp_plot': True})
    for chaser in C:
      chaser.plotStateHistory(params = {'sep_plots': False, 'disp_plot': True})
  else:
    S = [T]
    S.extend(C)
    indStateHistory = []
    for spacecraft in S:
      indStateHistory.append(spacecraft.genIndStateHistory())
    if orbit_sim:
      simX = []
      simY = []
      simZ = []
      for spacecraft in S[1:]:
        simX.append(indStateHistory[spacecraft.id][0])
        simY.append(indStateHistory[spacecraft.id][1])
        simZ.append(indStateHistory[spacecraft.id][2])
      fName = filenameCreator('../plots/', '.html')
      simulate(fName, simX, simY, simZ, [(agent+1) for agent in range(len(S)-1)], len(S)-1)
    plt.figure(figsize=(16,9))
    for spacecraft in S:
      cmap = plt.cm.get_cmap(plt.cm.viridis, 256)
      timeSeq = [step*Spacecraft.dt for step in range(len(indStateHistory[spacecraft.id][0]))]
      plt.subplot(int(len(S)/3),3,spacecraft.id+1)
      for i in range(spacecraft.numStates):
        label = '$x_'+str(i)+'$'
        plt.plot(timeSeq, indStateHistory[spacecraft.id][i], c = cmap(96*i), label = label)
      plt.legend()
      plt.ylabel('State History')
      plt.xlabel('Time')
    plt.show()