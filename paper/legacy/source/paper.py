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
import sys
import time
import numpy      as     np
from   gekko      import GEKKO
from   matplotlib import pyplot as plt

import params     as     p
from dynamics     import orbital_ellp_undrag

def runOrbit(system, timeSeq, x_0, eccentricity = 0.0, alpha = 0.0, beta = 0.0, *args, **kwargs):
  matrices, _, _ = system(dt = timeSeq[1] - timeSeq[0], n  = p.n,
                          eccentricity = eccentricity,
                          alpha        = alpha,
                          beta         = beta,)
  A, _, _, _, d  = matrices
  
  ## Initialize MPC ##
  m      = GEKKO(remote = False)
  m.time = timeSeq
  states = []
  
  ## Start Anomaly ##
  t_s = timeSeq[0]
  ## Final Anomaly ##
  t_f = timeSeq[-1]
  
  ## Initialize Variables ##
  if True:
    t =  m.Var   (value =      0)
    q =  m.Var   (value =      0, fixed_initial = False)
    x = [m.Var   (value = x_0[i], fixed_initial =  True) for i in range(len(x_0))]
  
  ## Constraint Equations ##
  eqs = []
  ### Time and Anomaly Update ###
  if True:
    eqs.append(t.dt() == 1)
    E = m.Intermediate (2*m.atan(np.sqrt((1-eccentricity)/(1+eccentricity))*m.tan(t/2)))
    M = m.Intermediate (E - eccentricity*m.sin(E))
    eqs.append(q      == p.t_p + t_s + M/p.n)
  ### System Dynamics ###
  for i in range(0,3):
    eqs.append(x[i+0].dt() == x[i+3])
    eqs.append(x[i+3].dt() == np.matmul(A(t + t_s, p.t_p, m = m)[i+3], x)
                              +                             0.0
                              +         d(t + t_s, p.t_p, m = m)[i+3])
  
  ## Solver Parameters ##
  if True:
    eqs = m.Equations(eqs)
    # m.options.OTOL       =    1e-7
    # m.options.RTOL       =    1e-7
    m.options.IMODE      =    6
    m.options.REDUCE     =    3
    m.options.SOLVER     =    3
    m.options.MAX_ITER   = 3000
    # m.options.DIAGLEVEL  =    0
    m.options.MAX_MEMORY =  512
    # m.options.COLDSTART  =    0
    # m.options.TIME_SHIFT =    0
  
  ## Solve MPC ##
  startTime = time.time()
  try:
    m.solve(disp = False)
  except:
    return 1, np.array([])
  stopTime = time.time()

  ## Print MPC Info ##
  if True:
    print(" Solver Time          : ", stopTime - startTime)
    print(" Solver Objective     : ", m.options.objfcnval)
    print(" Solver Status        : ", m.options.APPSTATUS)
    print(" Solver Internal Time : ", m.options.SOLVETIME)
    print()

  ## Result Formatting and Truncation ##
  if True:
    x_t = np.transpose([x[i].value for i in range(len(x))])
    states.extend(x_t)
  
  ## Clear Loop ##
  m.cleanup()
  del m

  ## Return Results ##
  return 0, np.array(states)

def calcEccentricityError(eccentricity, t_0, t_f, dt, x_0):
  print('~~~ Calculating Error for Eccentricity = ', eccentricity, ' ~~~')
  numSteps    = int((t_f-t_0)/dt)
  timeSeq     = np.linspace(t_0, t_f, numSteps)
  print('~ Simulation of an assumed Circular   Orbit ~')
  output_circ = runOrbit(system       = orbital_ellp_undrag,
                         timeSeq      = timeSeq,
                         x_0          = x_0,
                         eccentricity = 0.0,)
  print('~ Simulation of the actual Elliptical Orbit ~')
  output_ellp = runOrbit(system       = orbital_ellp_undrag,
                         timeSeq      = timeSeq,
                         x_0          = x_0,
                         eccentricity = eccentricity,)
  flag_circ, states_circ = output_circ
  flag_ellp, states_ellp = output_ellp
  if flag_circ or flag_ellp:
    raise Exception(RuntimeError, 'Solver failed to compute states')
  states_diff = np.subtract(states_ellp, states_circ)
  states_err  = np.array([np.linalg.norm(state_diff) for state_diff in states_diff])
  fState_diff = states_diff[-1]
  fState_err  = states_err [-1]
  fState_nerr = fState_err / np.linalg.norm(x_0)
  # return finStateErr, states_circ, states_ellp
  print('Difference       in Final States : ', fState_diff)
  print('Error            in Final States = ', fState_err )
  print('Normalized Error in Final States = ', fState_nerr)
  plt.figure()
  plt.plot(states_err)
  fig, axs = plt.subplots(2, 3)
  for der in range(2):
    for dim in range(3):
      axs[der][dim].plot(states_circ[:,der*3+dim], 'b-')
      axs[der][dim].plot(states_ellp[:,der*3+dim], 'r-')
  plt.show()

def calcDragError(eccentricity, alpha, beta, t_0, t_f, dt, x_0):
  print()
  print()
  print('~~~ Calculating Error for Drag Constant, alpha = ', alpha, ' for beta = ', beta, ' ~~~')
  numSteps    = int((t_f-t_0)/dt)
  timeSeq     = np.linspace(t_0, t_f, numSteps)
  print('~ Simulation of an assumed No-Drag Orbit ~')
  output_perf = runOrbit(system       = orbital_ellp_undrag,
                         timeSeq      = timeSeq,
                         x_0          = x_0,
                         eccentricity = eccentricity,
                         alpha        = 0.0,
                         beta         = beta,)
  print('~ Simulation of the actual Lossy   Orbit ~')
  output_drag = runOrbit(system       = orbital_ellp_undrag,
                         timeSeq      = timeSeq,
                         x_0          = x_0,
                         eccentricity = eccentricity,
                         alpha        = alpha,
                         beta         = beta,)
  flag_perf, states_perf = output_perf
  flag_drag, states_drag = output_drag
  if flag_perf or flag_drag:
    raise Exception(RuntimeError, 'Solver failed to compute states')
  states_diff = np.subtract(states_drag, states_perf)
  states_err  = np.array([np.linalg.norm(state_diff) for state_diff in states_diff])
  fState_diff = states_diff[-1]
  fState_err  = states_err [-1]
  fState_nerr = fState_err / np.linalg.norm(x_0)
  # return finStateErr, states_perf, states_drag
  print('Difference       in Final States : ', fState_diff)
  print('Error            in Final States = ', fState_err )
  print('Normalized Error in Final States = ', fState_nerr)
  plt.figure()
  plt.plot(states_err)
  fig, axs = plt.subplots(2, 3)
  for der in range(2):
    for dim in range(3):
      axs[der][dim].plot(states_perf[:,der*3+dim], 'b-')
      axs[der][dim].plot(states_drag[:,der*3+dim], 'r-')
  plt.show()

def plotAdaptorFile(folderName, fileName, fileNameExtension, data_length, dock_index):
  try:
    data = np.load(folderName + fileName + fileNameExtension, allow_pickle = True)
  except:
    print("File not found at ", folderName + fileName + fileNameExtension)
    return 1
  if fileNameExtension == '.npy.npz':
    main_estim_lists = data['main_estim_lists']
    main_range_lists = data['main_range_lists']
    actOrbitParams   = data['actOrbitParams']
    actOrbitParams   = actOrbitParams[()]
  elif fileNameExtension == '.npy':
    main_estim_lists, main_range_lists, actOrbitParams = data
  actOrbitParams   = [actOrbitParams['eccentricity'], actOrbitParams['drag_alpha'], actOrbitParams['drag_beta']]
  if len(main_estim_lists[0]) == 0:
    print(len(main_estim_lists[0]))
    return 1
  
  data_length = min(data_length, len(main_estim_lists[0]))
  dock_index = min(dock_index, len(main_estim_lists[0])-1)
  print( [main_estim_lists[param][dock_index] for param in [0,1,2]])
  print([[main_range_lists[param][bound][dock_index] for bound in [0,1]] for param in [0,1,2]])
  print(actOrbitParams)
  
  for param in range(3):
    plt.figure(figsize=(6,4))
    plt.plot(main_estim_lists[param]   [:data_length], 'g-',  label = 'Estimated Value')
    plt.plot(main_range_lists[param][0][:data_length], 'b-',  label = 'Lower Bound')
    plt.plot(main_range_lists[param][1][:data_length], 'r-',  label = 'Upper Bound')
    plt.plot([ actOrbitParams[param]]   *data_length,  'k--', label = 'Actual Value')
    plt.xlabel('Data Index', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    leg = plt.legend()
    leg_lines = leg.get_lines()
    leg_texts = leg.get_texts()
    plt.setp(leg_lines, linewidth=4)
    plt.setp(leg_texts, fontsize='x-large')
    plt.grid()
    plt.tight_layout()
    plt.show()

def plotMPCFile(folderName, fileName, fileNameExtension):
  try:
    data = np.load(folderName + fileName + fileNameExtension, allow_pickle = True)
  except:
    print("File not found at ", folderName + fileName + fileNameExtension)
    return 1
  if fileNameExtension == '.npy.npz':
    main_act_states = data['main_act_states']
    main_act_inputs = data['main_act_inputs']
    main_nom_states = data['main_nom_states']
    main_nom_inputs = data['main_nom_inputs']
    main_fin_states = data['main_fin_states']
    main_tgt_states = data['main_tgt_states']
    main_target_thetas = data['main_target_thetas']
    main_X_f = data['main_X_f']
  elif fileNameExtension == '.npy':
    main_act_states, main_act_inputs, main_nom_states, main_nom_inputs, main_fin_states, main_tgt_states, main_target_thetas, main_X_f = data

  # data_length = 1559
  data_length = int(len(main_act_states))
  print("Data Length: ", data_length)
  timeSeq = np.linspace(0, data_length*p.dt, data_length)
  main_act_states = main_act_states[:data_length]
  main_act_inputs = main_act_inputs[:data_length]
  main_nom_states = main_nom_states[:data_length]
  main_nom_inputs = main_nom_inputs[:data_length]
  main_fin_states = main_fin_states[:data_length]
  main_tgt_states = main_tgt_states[:data_length]
  main_target_thetas = main_target_thetas[:data_length]
  main_X_f = main_X_f[:data_length]
  norm_act_states = np.array([np.linalg.norm(main_act_states[i][:3]-main_tgt_states[i]) for i in range(len(main_act_states))])

  p_tilde = np.array([np.linalg.norm(main_act_states[i][:3]-main_nom_states[i][:3]) for i in range(len(main_act_states))])
  for i in range(len(main_act_states)):
    prev_rad = int(i/20)*20 -  1
    next_rad = int(i/20)*20 + 19
    # if i%20 != 19:
    #   p_tilde[i] = p_tilde[i-1] + (p_tilde[i] - p_tilde[i-1])/(20)
    try:
      p_tilde[i] = p_tilde[prev_rad] + (p_tilde[next_rad] - p_tilde[prev_rad])/(20)*(i%20)
    except:
      p_tilde[i] = p_tilde[prev_rad]
  main_tgt_states = [main_tgt_states[i]*(1 + 2.0*p_tilde[i]) for i in range(len(main_tgt_states))]

  dock_index = next(x[0] for x in enumerate(norm_act_states) if x[1] < 1.5)
  print("Min Norm of Actual States: ", np.min(norm_act_states))
  print("Min Norm is at Index:      ", np.argmin(norm_act_states))
  print("Norm at Final State:       ", norm_act_states[-1])
  print("Dock Index:                ", dock_index)
  print()

  if 'ADTMPC' in fileName or 'AMPC' in fileName:
    plotAdaptorFile(folderName, fileName[:-3]+'adaptor', fileNameExtension, data_length = data_length, dock_index = dock_index)

  tubePlot = True
  if not tubePlot:
    for dim in range(6):
      plt.figure(figsize=(6,4))
      plt.xlabel('True Anomaly of the Orbit (rad)', fontsize=14)
      plt.ylabel('Position (m)', fontsize=14)
      plt.plot(timeSeq, [state[dim] for state in main_act_states], 'b-', label = 'Actual States')
      plt.plot(timeSeq, [state[dim] for state in main_fin_states], 'g-', label = 'Final States')
      if dim < 3:
        plt.plot(timeSeq, [state[dim] for state in main_tgt_states], 'k-', label = 'Target States')
      leg = plt.legend()
      leg_lines = leg.get_lines()
      leg_texts = leg.get_texts()
      plt.setp(leg_lines, linewidth=4)
      plt.setp(leg_texts, fontsize='x-large')
      plt.tight_layout()
      plt.show()
    if dock_index > 0:
      for dim in range(6):
        plt.figure(figsize=(6,4))
        plt.xlabel('True Anomaly of the Orbit (rad)', fontsize=14)
        plt.ylabel('Position (m)', fontsize=14)
        plt.plot(timeSeq[dock_index:], [state[dim] for state in main_act_states[dock_index:]], 'b-', label = 'Actual States')
        plt.plot(timeSeq[dock_index:], [state[dim] for state in main_fin_states[dock_index:]], 'g-', label = 'Final States')
        if dim < 3:
          plt.plot(timeSeq[dock_index:], [state[dim] for state in main_tgt_states[dock_index:]], 'k-', label = 'Target States')
        leg = plt.legend()
        leg_lines = leg.get_lines()
        leg_texts = leg.get_texts()
        plt.setp(leg_lines, linewidth=4)
        plt.setp(leg_texts, fontsize='x-large')
        plt.tight_layout()
        plt.show()
  else:
    plt.figure(figsize=(6,4))
    plt.xlabel('True Anomaly of the Orbit (rad)', fontsize=14)
    plt.ylabel('Positional Norm (m)', fontsize=14)
    plt.plot(timeSeq, [np.linalg.norm(main_act_states[i][:3]-main_tgt_states[i]) for i in range(len(main_act_states))], 'b-', label = 'Actual States')
    plt.plot(timeSeq, [np.linalg.norm(main_act_states[i][:3]-main_tgt_states[i]-p_tilde[i]) for i in range(len(main_act_states))], 'r--', label = 'Tube Boundary')
    plt.plot(timeSeq, [np.linalg.norm(main_act_states[i][:3]-main_tgt_states[i]+p_tilde[i]) for i in range(len(main_act_states))], 'r--')
    leg = plt.legend()
    leg_lines = leg.get_lines()
    leg_texts = leg.get_texts()
    plt.setp(leg_lines, linewidth=4)
    plt.setp(leg_texts, fontsize='x-large')
    plt.grid()
    plt.tight_layout()
    plt.show()

  return 0

def plotDeflectFile(folderName, fileName, fileNameExtension):
  try:
    data = np.load(folderName + fileName + fileNameExtension, allow_pickle = True)
  except:
    print("File not found at ", folderName + fileName + fileNameExtension)
    return 1
  if fileNameExtension == '.npy.npz':
    T = data['T']
    angles = data['angles']
    x_f_body = data['x_f_body']
    forces = data['forces']
  elif fileNameExtension == '.npy':
    T, angles, x_f_body, forces = data
  
  print('Position of the Chaser: ', x_f_body)
  print('Initial Angular Position: [', angles[0][ 0], angles[1][ 0], angles[2][ 0], ']')
  print('Initial Angular Velocity: [', angles[3][ 0], angles[4][ 0], angles[5][ 0], ']')
  print('Final   Angular Position: [', angles[0][-1], angles[1][-1], angles[2][-1], ']')
  print('Final   Angular Velocity: [', angles[3][-1], angles[4][-1], angles[5][-1], ']')
  print('Norm of Initial Angular Velocity: ', np.linalg.norm([angles[i][ 0] for i in range(3,6)]))
  print('Norm of Final   Angular Velocity: ', np.linalg.norm([angles[i][-1] for i in range(3,6)]))
  
  forces = forces[0]
  angPos = angles[0:3]
  angVel = angles[3:6]
  
  timeSeq = np.linspace(0, len(angles[0])*p.dt, len(angles[0]))
  plt.figure(figsize=(6,3))
  plt.title('Angular Position')
  for dim in range(3):
    plt.plot(timeSeq, angPos[dim], label = ['x', 'y', 'z'][dim])
  plt.xlabel('True Anomaly of the Orbit (rad)', fontsize=14)
  plt.ylabel('Angular Position (rad)', fontsize=14)
  leg = plt.legend()
  leg_lines = leg.get_lines()
  leg_texts = leg.get_texts()
  plt.setp(leg_lines, linewidth=4)
  plt.setp(leg_texts, fontsize='x-large')
  plt.tight_layout()
  plt.show()

  plt.figure(figsize=(6,3))
  plt.title('Angular Velocity')
  for dim in range(3):
    plt.plot(timeSeq, angVel[dim], label = ['x', 'y', 'z'][dim])
  plt.xlabel('True Anomaly of the Orbit (rad)', fontsize=14)
  plt.ylabel('Angular Velocity (rad/s)', fontsize=14)
  leg = plt.legend()
  leg_lines = leg.get_lines()
  leg_texts = leg.get_texts()
  plt.setp(leg_lines, linewidth=4)
  plt.setp(leg_texts, fontsize='x-large')
  plt.tight_layout()
  plt.show()

  plt.figure(figsize=(6,3))
  plt.title('Forces')
  for dim in range(3):
    plt.plot(timeSeq, [val/50 for val in forces[dim]], label = ['x', 'y', 'z'][dim])
  plt.xlabel('True Anomaly of the Orbit (rad)', fontsize=14)
  plt.ylabel('Force (N)', fontsize=14)
  leg = plt.legend()
  leg_lines = leg.get_lines()
  leg_texts = leg.get_texts()
  plt.setp(leg_lines, linewidth=4)
  plt.setp(leg_texts, fontsize='x-large')
  plt.tight_layout()
  plt.show()

def plotConstraints(folderName, fileName, fileNameExtension):
  try:
    data = np.load(folderName + fileName + fileNameExtension, allow_pickle = True)
  except:
    print("File not found at ", folderName + fileName + fileNameExtension)
    return 1
  if fileNameExtension == '.npy.npz':
    main_act_states = data['main_act_states']
    main_act_inputs = data['main_act_inputs']
    main_nom_states = data['main_nom_states']
    main_nom_inputs = data['main_nom_inputs']
    main_fin_states = data['main_fin_states']
    main_tgt_states = data['main_tgt_states']
    main_target_thetas = data['main_target_thetas']
    main_X_f = data['main_X_f']
  elif fileNameExtension == '.npy':
    main_act_states, main_act_inputs, main_nom_states, main_nom_inputs, main_fin_states, main_tgt_states, main_target_thetas, main_X_f = data
  
  norm_act_states = np.array([np.linalg.norm(main_act_states[i][:3]-main_tgt_states[i]) for i in range(len(main_act_states))])
  dock_index = next(x[0] for x in enumerate(norm_act_states) if x[1] < 1.5)
  timeSeq = np.linspace(0, len(main_act_states)*p.dt, len(main_act_states))
  
  start_index = 200
  final_index = 750
  timeSeq = np.array(timeSeq[start_index:final_index])
  main_act_states = np.array(main_act_states[start_index:final_index])
  main_act_inputs = np.array(main_act_inputs[start_index:final_index])
  main_nom_states = np.array(main_nom_states[start_index:final_index])
  main_nom_inputs = np.array(main_nom_inputs[start_index:final_index])
  main_fin_states = np.array(main_fin_states[start_index:final_index])
  main_tgt_states = np.array(main_tgt_states[start_index:final_index])
  main_target_thetas = np.array(main_target_thetas[start_index:final_index])
  main_X_f = np.array(main_X_f[start_index:final_index])

  def split_states(states, idx):
    return [state[idx] for state in states]
  
  def split_inputs(inputs, idx):
    return [input[idx] for input in inputs]
  
  def compute_norms(states):
    return [np.linalg.norm(state) for state in states]
  
  from commons import tait_bryan_to_rotation_matrix
  
  sXa, sYa, sZa = [split_states(main_act_states, idx) for idx in range(3)]
  con1, con2, con12, con3 = [], [], [], []
  for t_iter, time in enumerate(timeSeq):
    rotMatrix = tait_bryan_to_rotation_matrix(main_target_thetas[t_iter])
    sa = np.array([sXa[t_iter], sYa[t_iter], sZa[t_iter]])
    sa = np.matmul(rotMatrix.T, sa)
    con1.append(sa[0]**2 + sa[1]**2 - p.targetLimit['r_T']**2)
    con2.append(sa[2]**2 - p.targetLimit['l_T']**2)
    con12.append(np.max([con1[-1], con2[-1]]) < 0.00)
    con3.append(sa[0]**2 + sa[1]**2 + sa[2]**2)
  plt.figure(figsize=(6,4))
  plt.plot(timeSeq, con1, 'b-', label = 'Constraint 1: $s_X^2+s_Y^2-r_T^2$')
  plt.plot(timeSeq, con2, 'g-', label = 'Constraint 2: $s_Z^2-l_T^2$')
  # plt.fill_between(timeSeq,
  #                  max(np.max(con1), np.max(con2)),
  #                  min(np.min(con1), np.min(con2)),
  #                  where = np.array(con12),
  #                  alpha = 0.5, color = 'r', label = 'Violation of Constraints 1 and 2')
  plt.xlabel('True Anomaly of the Orbit (rad)', fontsize=14)
  plt.ylabel('Position (m)', fontsize=14)
  leg = plt.legend(loc = 'lower right')
  leg_lines = leg.get_lines()
  leg_texts = leg.get_texts()
  plt.setp(leg_lines, linewidth=4)
  plt.setp(leg_texts, fontsize='x-large')
  plt.grid()
  plt.tight_layout()
  plt.show()

def plotFromOrbitSim(folderName, fileName, fileNameExtension):
  try:
    data = np.load(folderName + fileName + fileNameExtension, allow_pickle = True)
  except:
    print("File not found at ", folderName + fileName + fileNameExtension)
    return 1
  if fileNameExtension == '.npy.npz':
    main_act_states = data['main_act_states']
    main_act_inputs = data['main_act_inputs']
    main_nom_states = data['main_nom_states']
    main_nom_inputs = data['main_nom_inputs']
    main_fin_states = data['main_fin_states']
    main_tgt_states = data['main_tgt_states']
    main_target_thetas = data['main_target_thetas']
    main_X_f = data['main_X_f']
  elif fileNameExtension == '.npy':
    main_act_states, main_act_inputs, main_nom_states, main_nom_inputs, main_fin_states, main_tgt_states, main_target_thetas, main_X_f = data
  norm_act_states = np.array([np.linalg.norm(main_act_states[i][:3]-main_tgt_states[i]) for i in range(len(main_act_states))])
  dock_index = next(x[0] for x in enumerate(norm_act_states) if x[1] < 1.5)
  
  from orbitsim import mpc_plot
  mpc_plot_kwargs = {'act_states' : main_act_states,
                     'act_inputs' : main_act_inputs,
                     'nom_states' : main_nom_states,
                     'nom_inputs' : main_nom_inputs,
                     'fin_states' : main_fin_states,
                     'tgt_states' : main_tgt_states,
                     'x_f_list'   : main_X_f,
                     'dt'         : p.dt,
                     'plotFlags'  : {'plot_act'    :  True,
                                     'plot_act_sim':  True,
                                     'plot_act_con':  True,
                                     'plot_nom'    :  True,
                                     'plot_nom_sim':  True,
                                     'plot_nom_con':  True,},
                     'target_thetas' : main_target_thetas,
                     'dock_index'    : dock_index}
  mpc_plot(**mpc_plot_kwargs)

if __name__ == '__main__':
  mainSeq = int(sys.argv[1])
  if mainSeq == 1:
    calcEccentricityError(eccentricity = 0.125,
                          t_0 = 0.0, t_f = (np.pi/2)/2,
                          dt  = (np.pi/2)/400,
                          x_0 = np.array([10.00, 2.00, 5.00,
                                           1.00, 0.50, 0.50,]))
    calcDragError(eccentricity = 0.125,
                  alpha        = 1.300e-7,
                  beta         = 2.600e-7,
                  t_0 = 0.0, t_f = (np.pi/2)/2,
                  dt  = (np.pi/2)/400,
                  x_0 = np.array([10.00, 2.00, 5.00,
                                   1.00, 0.50, 0.50,]))
  if mainSeq == 2:
    session = str(int(sys.argv[2]))
    scene = str(int(sys.argv[3]))
    opType = 'Single-Agent Rendezvous and Docking'
    if int(session) <= 2:
      extension = '.npy'
    else:
      extension = '.npy.npz'
    plotMPCFile('../plots/Tests/Session 0'+session+'/Scene 0'+scene+'/Test 01 - Complete Pipeline - ADTMPC - '+opType+'/', 'mpc_test_ADTMPC_mpc', extension)
    plotMPCFile('../plots/Tests/Session 0'+session+'/Scene 0'+scene+'/Test 02 - Complete Pipeline - DTMPC - '+opType+'/', 'mpc_test_DTMPC_mpc', extension)
    plotMPCFile('../plots/Tests/Session 0'+session+'/Scene 0'+scene+'/Test 03 - Complete Pipeline - AMPC - '+opType+'/', 'mpc_test_AMPC_mpc', extension)
    plotMPCFile('../plots/Tests/Session 0'+session+'/Scene 0'+scene+'/Test 04 - Complete Pipeline - MPC - '+opType+'/', 'mpc_test_MPC_mpc', extension)
  if mainSeq == 3:
    session = str(int(sys.argv[2]))
    scene = str(int(sys.argv[3]))
    opType = 'Single-Agent Rendezvous'
    if int(session) <= 2:
      extension = '.npy'
    else:
      extension = '.npy.npz'
    plotDeflectFile('../plots/Tests/Session 0'+session+'/Scene 0'+scene+'/Test 01 - Complete Pipeline - ADTMPC - '+opType+'/', 'mpc_test_ADTMPC_deflection', extension)
  if mainSeq == 4:
    session = str(int(sys.argv[2]))
    scene = str(int(sys.argv[3]))
    opType = 'Single-Agent Docking'
    if int(session) <= 2:
      extension = '.npy'
    else:
      extension = '.npy.npz'
    plotConstraints('../plots/Tests/Session 0'+session+'/Scene 0'+scene+'/Test 01 - Complete Pipeline - ADTMPC - '+opType+'/', 'mpc_test_ADTMPC_mpc', extension)
    plotConstraints('../plots/Tests/Session 0'+session+'/Scene 0'+scene+'/Test 02 - Complete Pipeline - DTMPC - '+opType+'/', 'mpc_test_DTMPC_mpc', extension)
    plotConstraints('../plots/Tests/Session 0'+session+'/Scene 0'+scene+'/Test 03 - Complete Pipeline - AMPC - '+opType+'/', 'mpc_test_AMPC_mpc', extension)
    plotConstraints('../plots/Tests/Session 0'+session+'/Scene 0'+scene+'/Test 04 - Complete Pipeline - MPC - '+opType+'/', 'mpc_test_MPC_mpc', extension)
  if mainSeq == 5:
    session = str(int(sys.argv[2]))
    scene = str(int(sys.argv[3]))
    opType = 'Single-Agent Docking'
    if int(session) <= 2:
      extension = '.npy'
    else:
      extension = '.npy.npz'
    plotFromOrbitSim('../plots/Tests/Session 0'+session+'/Scene 0'+scene+'/Test 01 - Complete Pipeline - ADTMPC - '+opType+'/', 'mpc_test_ADTMPC_mpc', extension)
    plotFromOrbitSim('../plots/Tests/Session 0'+session+'/Scene 0'+scene+'/Test 02 - Complete Pipeline - DTMPC - '+opType+'/', 'mpc_test_DTMPC_mpc', extension)
    plotFromOrbitSim('../plots/Tests/Session 0'+session+'/Scene 0'+scene+'/Test 03 - Complete Pipeline - AMPC - '+opType+'/', 'mpc_test_AMPC_mpc', extension)
    plotFromOrbitSim('../plots/Tests/Session 0'+session+'/Scene 0'+scene+'/Test 04 - Complete Pipeline - MPC - '+opType+'/', 'mpc_test_MPC_mpc', extension)