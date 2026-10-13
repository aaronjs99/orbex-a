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
import random
import numpy as np
import plotly.offline       as ptyplt
import plotly.graph_objects as go
import plotly.express       as px
import matplotlib.pyplot    as plt
from   mpl_toolkits.mplot3d import Axes3D
import params as p
from commons import filenameCreator, latestDataFile, loadData, tait_bryan_to_rotation_matrix

random.seed(0)

# FUNCTION DEFINITIONS
### Plotter for mpc.py
def mpc_plot(act_states, act_inputs, nom_states, nom_inputs, fin_states, tgt_states,
             x_f_list, dt, plotFlags, target_thetas, *args, **kwargs):
  if 'fName_sim' not in kwargs:
    fName_sim = '../plots/mpc_test'
  else:
    fName_sim = kwargs['fLoc'] + kwargs['fName_sim']
  
  act_states = np.array(act_states)
  act_inputs = np.array(act_inputs)
  nom_states = np.array(nom_states)
  nom_inputs = np.array(nom_inputs)
  fin_states = np.array(fin_states)
  tgt_states = np.array(tgt_states)

  def split_states(states, idx):
    return [state[idx] for state in states]
  
  def split_inputs(inputs, idx):
    return [input[idx] for input in inputs]
  
  def compute_norms(states):
    return [np.linalg.norm(state) for state in states]
  
  def print_min_max(values, label):
    print(label + ': Min =', np.min(values), ', Max =', np.max(values))

  print(len(act_states), len(act_inputs), len(nom_states), len(nom_inputs), len(fin_states), len(tgt_states))
  ### Split Actual States into Lists ###
  sXa, sYa, sZa, vXa, vYa, vZa = [split_states(act_states, idx) for idx in range(6)]
  uXa, uYa, uZa = [split_inputs(act_inputs, idx) for idx in range(3)]
  sNa, vNa, uNa = [compute_norms(act_states[:, :3]), compute_norms(act_states[:, 3:]), compute_norms(act_inputs)]
  ### Split Nominal States into Lists ###
  sXn, sYn, sZn, vXn, vYn, vZn = [split_states(nom_states, idx) for idx in range(6)]
  uXn, uYn, uZn = [split_inputs(nom_inputs, idx) for idx in range(3)]
  sNn, vNn, uNn = [compute_norms(nom_states[:, :3]), compute_norms(nom_states[:, 3:]), compute_norms(nom_inputs)]
  ### Split Final States into Lists ###
  sXf, sYf, sZf = [split_states(fin_states, idx) for idx in range(3)]
  ### Split Target States into Lists ###
  sXt, sYt, sZt = [split_states(tgt_states, idx) for idx in range(3)]

  t = np.linspace(p.t_p, p.t_p + (len(sXa) - 1)*dt, len(sXa))
  ### Gemerate Values for Only Docking ###
  try:
    dock_index = kwargs['dock_index']
  except:
    try:
      dock_index = np.nonzero(sXf)[0][0]  
    except:
      dock_index = -1
  if dock_index >= 0:
    t_dock   = t  [dock_index:]
    sXa_dock = sXa[dock_index:]
    sYa_dock = sYa[dock_index:]
    sZa_dock = sZa[dock_index:]
    vXa_dock = vXa[dock_index:]
    vYa_dock = vYa[dock_index:]
    vZa_dock = vZa[dock_index:]
    uXa_dock = uXa[dock_index:]
    uYa_dock = uYa[dock_index:]
    uZa_dock = uZa[dock_index:]
    sNa_dock = sNa[dock_index:]
    vNa_dock = vNa[dock_index:]
    uNa_dock = uNa[dock_index:]
    sXn_dock = sXn[dock_index:]
    sYn_dock = sYn[dock_index:]
    sZn_dock = sZn[dock_index:]
    vXn_dock = vXn[dock_index:]
    vYn_dock = vYn[dock_index:]
    vZn_dock = vZn[dock_index:]
    uXn_dock = uXn[dock_index:]
    uYn_dock = uYn[dock_index:]
    uZn_dock = uZn[dock_index:]
    sNn_dock = sNn[dock_index:]
    vNn_dock = vNn[dock_index:]
    uNn_dock = uNn[dock_index:]
    sXf_dock = sXf[dock_index:]
    sYf_dock = sYf[dock_index:]
    sZf_dock = sZf[dock_index:]
    sXt_dock = sXt[dock_index:]
    sYt_dock = sYt[dock_index:]
    sZt_dock = sZt[dock_index:]

  ### Print Minimum and Maximum Values for Chaser Distance, Velocity, and Thrust Input ###
  if len(sNa) > 0:
    print()
    print('~~~ Actual States ~~~')
    print_min_max(sNa, 'Chaser Distance ')
    print_min_max(vNa, 'Chaser Velocity ')
    print_min_max(uNa, 'Chaser Input    ')
    print('~~~ Generated States ~~~')
    print_min_max(sNn, 'Chaser Distance ')
    print_min_max(vNn, 'Chaser Velocity ')
    print_min_max(uNn, 'Chaser Input    ')

  ### Generate Labels ###
  if True:
    labelX  = [    '$s_X$',     '$s_Y$',     '$s_Z$',
                   '$v_X$',     '$v_Y$',     '$v_Z$',]
    labelX1 = ['$s_{X,F}$', '$s_{Y,F}$', '$s_{Z,F}$',]
    labelX2 = ['$s_{X,T}$', '$s_{Y,T}$', '$s_{Z,T}$',]
    labelU  = [    '$u_X$',     '$u_Y$',     '$u_Z$',]
    labelY  = [  '$||s||$',   '$||v||$',   '$||u||$',]

  ### Plot Actual States ###
  if plotFlags['plot_act']:
    if plotFlags['plot_act_sim']:
      simulate(fName_sim + '_act.html', [sXa,],
                      [sYa,],
                      [sZa,], [0], 1,
               pointList = x_f_list,)
              #  shape = [{'type'    : 'sphere',
              #            'radius'  : p.radialLimit,
              #            'center'  : [0, 0, 0],
              #            'opacity' :  0.25,},],)
    plotter_kwargs = {'x'  : [sXa, sYa, sZa,
                              vXa, vYa, vZa,],
                      'u'  : [uXa, uYa, uZa,],
                      'y'  : [sNa, vNa, uNa,],
                      'x1' : [sXf, sYf, sZf,],
                      'x2' : [sXt, sYt, sZt,],
                      'labelX'  : labelX,
                      'labelU'  : labelU,
                      'labelY'  : labelY,
                      'labelX1' : labelX1,
                      'labelX2' : labelX2,}
    if 'fLoc' in kwargs:
      plotter_kwargs['fName'] = kwargs['fLoc']+'act_states_all_time.png'
    plotter(t, **plotter_kwargs)

    ### Plot Actual States for Only Docking ###
    if dock_index >= 0:
      plotter_kwargs = {'x'  : [sXa_dock, sYa_dock, sZa_dock,
                                vXa_dock, vYa_dock, vZa_dock,],
                        'u'  : [uXa_dock, uYa_dock, uZa_dock,],
                        'y'  : [sNa_dock, vNa_dock, uNa_dock,],
                        'x1' : [sXf_dock, sYf_dock, sZf_dock,],
                        'x2' : [sXt_dock, sYt_dock, sZt_dock,],
                        'labelX'  : labelX,
                        'labelU'  : labelU,
                        'labelY'  : labelY,
                        'labelX1' : labelX1,
                        'labelX2' : labelX2,}
      if 'fLoc' in kwargs:
        plotter_kwargs['fName'] = kwargs['fLoc']+'act_states_docking.png'
      plotter(t_dock, **plotter_kwargs)
  
  ### Plot Actual State Constraints ###
  if plotFlags['plot_act_con']:
    con1, con2, con12, con3 = [], [], [], []
    for t_iter, time in enumerate(t):
      rotMatrix = tait_bryan_to_rotation_matrix(target_thetas[t_iter])
      sa = np.array([sXa[t_iter], sYa[t_iter], sZa[t_iter]])
      sa = np.matmul(rotMatrix.T, sa)
      con1.append(sa[0]**2 + sa[1]**2 - p.targetLimit['r_T']**2)
      con2.append(sa[2]**2 - p.targetLimit['l_T']**2)
      con12.append(np.max([con1[-1], con2[-1]]) < 0.00)
      con3.append(sa[0]**2 + sa[1]**2 + sa[2]**2)
    plotter_kwargs = {'x'  : [sXa, sYa, sZa,
                              vXa, vYa, vZa,],
                      'u'  : [con1, con2, con12,],
                      'x1' : [sXf, sYf, sZf,],
                      'x2' : [sXt, sYt, sZt,],
                      'labelX'  : labelX,
                      'labelU'  : ['Constraint 1: $s_X^2+s_Y^2-r_T^2$',
                                   'Constraint 2: $s_Z^2-l_T^2$',
                                   'Violation of Constraints 1 and 2',],
                                   # '$s_X^2+s_Y^2+s_Z^2$',],
                      'labelX1' : labelX1,
                      'labelX2' : labelX2,}
    if 'fLoc' in kwargs:
      plotter_kwargs['fName'] = kwargs['fLoc']+'constraints_act_states_all_time.png'
    plotter(t, **plotter_kwargs)
    
    ### Plot Actual States for Only Docking ###
    if dock_index >= 0:
      con1_dock  = con1 [dock_index:]
      con2_dock  = con2 [dock_index:]
      con12_dock = con12[dock_index:]
      con3_dock  = con3 [dock_index:]
      plotter_kwargs = {'x'  : [sXa_dock, sYa_dock, sZa_dock,
                                vXa_dock, vYa_dock, vZa_dock,],
                        'u'  : [con1_dock, con2_dock, con12_dock,],
                        'x1' : [sXf_dock, sYf_dock, sZf_dock,],
                        'x2' : [sXt_dock, sYt_dock, sZt_dock,],
                        'labelX'  : labelX,
                        'labelU'  : ['Constraint 1: $s_X^2+s_Y^2-r_T^2$',
                                     'Constraint 2: $s_Z^2-l_T^2$',
                                     'Violation of Constraints 1 and 2',],
                                     # '$s_X^2+s_Y^2+s_Z^2$',],
                        'labelX1' : labelX1,
                        'labelX2' : labelX2,}
      if 'fLoc' in kwargs:
        plotter_kwargs['fName'] = kwargs['fLoc']+'constraints_act_states_docking.png'
      plotter(t_dock, **plotter_kwargs)

  ### Plot Nominal States ###
  if plotFlags['plot_nom']:
    if plotFlags['plot_nom_sim']:
      simulate(fName_sim + '_nom.html', [sXn,],
                      [sYn,],
                      [sZn,], [0], 1,
               pointList = x_f_list,
               shape = [{'type'   : 'sphere',
                         'radius' : p.radialLimit,
                         'center' : [0, 0, 0],
                         'opacity': 0.25,},],)
    # plotter(t, x  = [sXn, sYn, sZn,
    #                  vXn, vYn, vZn,],
    #            u  = [uXn, uYn, uZn,],
    #            y  = [sNn, vNn, uNn,],
    #            labelX = labelX, labelU = labelU, labelY = labelY,
    #            x1 = [sXf, sYf, sZf,],
    #            x2 = [sXt, sYt, sZt,],
    #            labelX1 = labelX1, labelX2 = labelX2,)
    plotter_kwargs = {'x'  : [sXn, sYn, sZn,
                              vXn, vYn, vZn,],
                      'u'  : [uXn, uYn, uZn,],
                      'y'  : [sNn, vNn, uNn,],
                      'x1' : [sXf, sYf, sZf,],
                      'x2' : [sXt, sYt, sZt,],
                      'labelX'  : labelX,
                      'labelU'  : labelU,
                      'labelY'  : labelY,
                      'labelX1' : labelX1,
                      'labelX2' : labelX2,}
    if 'fLoc' in kwargs:
      plotter_kwargs['fName'] = kwargs['fLoc']+'nom_states_all_time.png'
    plotter(t, **plotter_kwargs)

    ### Plot Nominal States for Only Docking ###
    if dock_index >= 0:
      plotter_kwargs = {'x'  : [sXn_dock, sYn_dock, sZn_dock,
                                vXn_dock, vYn_dock, vZn_dock,],
                        'u'  : [uXn_dock, uYn_dock, uZn_dock,],
                        'y'  : [sNn_dock, vNn_dock, uNn_dock,],
                        'x1' : [sXf_dock, sYf_dock, sZf_dock,],
                        'x2' : [sXt_dock, sYt_dock, sZt_dock,],
                        'labelX'  : labelX,
                        'labelU'  : labelU,
                        'labelY'  : labelY,
                        'labelX1' : labelX1,
                        'labelX2' : labelX2,}
      if 'fLoc' in kwargs:
        plotter_kwargs['fName'] = kwargs['fLoc']+'nom_states_docking.png'
      plotter(t_dock, **plotter_kwargs)

  ### Plot Nominal State Constraints ###
  if plotFlags['plot_nom_con']:
    con1, con2, con12, con3 = [], [], [], []
    for t_iter, time in enumerate(t):
      rotMatrix = tait_bryan_to_rotation_matrix(target_thetas[t_iter])
      sn = np.array([sXn[t_iter], sYn[t_iter], sZn[t_iter]])
      sn = np.matmul(rotMatrix.T, sn)
      con1.append(sn[0]**2 + sn[1]**2 - p.targetLimit['r_T']**2)
      con2.append(sn[2]**2 - p.targetLimit['l_T']**2)
      con12.append(np.max([con1[-1], con2[-1]]) < 0)
      con3.append(sn[0]**2 + sn[1]**2 + sn[2]**2)
    # plotter(t, x  = [sXn, sYn, sZn,
    #                  vXn, vYn, vZn,],
    #            u  = [con1, con2, con12,],
    #            labelX = labelX, labelU = ['Constraint 1: $s_X^2+s_Y^2-r_T^2$',
    #                                       'Constraint 2: $s_Z^2-l_T^2$',
    #                                       'Violation of Constraints 1 and 2',],
    #                                       # '$s_X^2+s_Y^2+s_Z^2$',],
    #            x1 = [sXf, sYf, sZf,],
    #            x2 = [sXt, sYt, sZt,],
    #            labelX1 = labelX1, labelX2 = labelX2,)
    plotter_kwargs = {'x'  : [sXn, sYn, sZn,
                              vXn, vYn, vZn,],
                      'u'  : [con1, con2, con12,],
                      'x1' : [sXf, sYf, sZf,],
                      'x2' : [sXt, sYt, sZt,],
                      'labelX'  : labelX,
                      'labelU'  : ['Constraint 1: $s_X^2+s_Y^2-r_T^2$',
                                   'Constraint 2: $s_Z^2-l_T^2$',
                                   'Violation of Constraints 1 and 2',],
                                   # '$s_X^2+s_Y^2+s_Z^2$',],
                      'labelX1' : labelX1,
                      'labelX2' : labelX2,}
    if 'fLoc' in kwargs:
      plotter_kwargs['fName'] = kwargs['fLoc']+'constraints_nom_states_all_time.png'
    plotter(t, **plotter_kwargs)

    ### Plot Nominal States for Only Docking ###
    if dock_index >= 0:
      con1_dock  = con1 [dock_index:]
      con2_dock  = con2 [dock_index:]
      con12_dock = con12[dock_index:]
      con3_dock  = con3 [dock_index:]
      plotter_kwargs = {'x'  : [sXn_dock, sYn_dock, sZn_dock,
                                vXn_dock, vYn_dock, vZn_dock,],
                        'u'  : [con1_dock, con2_dock, con12_dock,],
                        'x1' : [sXf_dock, sYf_dock, sZf_dock,],
                        'x2' : [sXt_dock, sYt_dock, sZt_dock,],
                        'labelX'  : labelX,
                        'labelU'  : ['Constraint 1: $s_X^2+s_Y^2-r_T^2$',
                                     'Constraint 2: $s_Z^2-l_T^2$',
                                     'Violation of Constraints 1 and 2',],
                                     # '$s_X^2+s_Y^2+s_Z^2$',],
                        'labelX1' : labelX1,
                        'labelX2' : labelX2,}
      if 'fLoc' in kwargs:
        plotter_kwargs['fName'] = kwargs['fLoc']+'constraints_nom_states_docking.png'
      plotter(t_dock, **plotter_kwargs)

### Plotter for adaptor.py
def adaptor_plot(estim_lists, range_lists, orbitParams, *args, **kwargs):
  if 'rangeParams' in kwargs:
    rangeParams = kwargs['rangeParams']
    plt.figure(figsize = (10, 10))
    plt.plot(rangeParams['dt']*np.arange(rangeParams['data_range']), orbitParams['u_t'][0], 'b-')
    plt.plot(rangeParams['dt']*np.arange(rangeParams['data_range']), orbitParams['u_t'][1], 'g-')
    plt.plot(rangeParams['dt']*np.arange(rangeParams['data_range']), orbitParams['u_t'][2], 'r-')
    plt.show()

    if 'W' in kwargs:
      W = kwargs['W']
      plt.figure(figsize = (10, 10))
      plt.plot(rangeParams['dt']*np.arange(rangeParams['data_range']), W[:, 0], 'b-')
      plt.plot(rangeParams['dt']*np.arange(rangeParams['data_range']), W[:, 1], 'g-')
      plt.plot(rangeParams['dt']*np.arange(rangeParams['data_range']), W[:, 2], 'r-')
      plt.show()
  
  eEstim_list, aEstim_list, bEstim_list = estim_lists
  eRange_list, aRange_list, bRange_list = range_lists
  fig, axs = plt.subplots(3, 1, figsize = (10, 10))
  
  axs[0].plot(eEstim_list   , 'b-')
  axs[0].plot(eRange_list[0], 'r-')
  axs[0].plot(eRange_list[1], 'r-')
  axs[0].plot([orbitParams['eccentricity'] for i in range(len(eEstim_list))], 'r--')
  axs[0].set_title("Estimate of Eccentricity")

  axs[1].plot(aEstim_list   , 'b-')
  axs[1].plot(aRange_list[0], 'r-')
  axs[1].plot(aRange_list[1], 'r-')
  axs[1].plot([orbitParams['drag_alpha']   for i in range(len(aEstim_list))], 'r--')
  axs[1].set_title("Estimate of Alpha")

  axs[2].plot(bEstim_list   , 'b-')
  axs[2].plot(bRange_list[0], 'r-')
  axs[2].plot(bRange_list[1], 'r-')
  axs[2].plot([orbitParams['drag_beta']    for i in range(len(bEstim_list))], 'r--')
  axs[2].set_title("Estimate of Beta")

  if 'fLoc' in kwargs:
    plt.gcf().set_size_inches(10*plt.gcf().get_size_inches())
    plt.tight_layout()
    plt.savefig(kwargs['fLoc']+'param_est.png')
    plt.close()
  else:
    plt.show()

### Plotter for deflection.py
def deflection_plot(target, x, r, f, plotFlags, *args, **kwargs):
  ### Unpack Parameters ###
  dt               = kwargs   ['dt']
  numSteps         = kwargs   ['numSteps']
  numChasers       = kwargs   ['numChasers']
  shapeParams      = kwargs   ['shapeParams']
  targetShape      = kwargs   ['targetShape']
  initialTimeLapse = kwargs   ['initialTimeLapse']
  plot_target      = plotFlags['plot_target']
  plot_position    = plotFlags['plot_position']
  plot_force       = plotFlags['plot_force']
  
  ### Target Angular Position and Velocity Plots ###
  if plot_target:
    target_plot_kwargs = {'params' : {'sep_plots': False, 'disp_plot': False},}
    if 'fLoc' in kwargs:
      target_plot_kwargs['fLoc'] = kwargs['fLoc']
    target.plotStateHistory(**target_plot_kwargs)

  ### Chaser Position Plot ###
  if plot_position:
    plt.figure()
    ax = plt.axes(projection='3d')
    if   targetShape == 'cylinder':
      cylHeight = shapeParams['cylHeight']
      cylRadius = shapeParams['cylRadius']
      cylCenter = shapeParams['cylCenter']
      u, v = np.mgrid[0:2*np.pi:30j, -1.0:1.0:30j]
      x = cylCenter[0] + cylRadius * np.cos(u)
      y = cylCenter[1] + cylRadius * np.sin(u)
      z = cylCenter[2] + cylHeight * v
    elif targetShape == 'ellipsoid':
      ellRadX   = shapeParams['ellRadX']
      ellRadY   = shapeParams['ellRadY']
      ellRadZ   = shapeParams['ellRadZ']
      ellCenter = shapeParams['ellCenter']
      u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
      x = ellCenter[0] + ellRadX * np.cos(u) * np.sin(v)
      y = ellCenter[1] + ellRadY * np.sin(u) * np.sin(v)
      z = ellCenter[2] + ellRadZ * np.cos(v)
    ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r)

    for agent in range(numChasers):
      label = '$r_'+str(agent)+'$'
      state = r[agent]
      ax.scatter(state[0], state[1], state[2],
                label = label)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.legend()
    if 'fLoc' in kwargs:
      azims = [45*i for i in range(8)]
      for i in range(8):
        ax.view_init(45,azims[i])
        plt.gcf().set_size_inches(plt.gcf().get_size_inches())
        plt.tight_layout()
        plt.savefig(kwargs['fLoc']+'chaser_pos_ortho'+str(i+1)+'.png')
        ax.view_init( 0, azims[i])
        plt.gcf().set_size_inches(plt.gcf().get_size_inches())
        plt.tight_layout()
        plt.savefig(kwargs['fLoc']+'chaser_pos_vert'+str(i+1)+'.png')
        ax.view_init(88, azims[i])
        plt.gcf().set_size_inches(plt.gcf().get_size_inches())
        plt.tight_layout()
        plt.savefig(kwargs['fLoc']+'chaser_pos_top'+str(i+1)+'.png')
      plt.close()
    else:
      plt.show()
  
  ### Chaser Force Plot ###
  if plot_force:
    plt.figure()
    timeSeq = np.linspace(initialTimeLapse, initialTimeLapse + (numSteps-1)*dt, numSteps)
    for agent in range(numChasers):
      force = f[agent]
      for j in range(3):
        label = '$f_'+str(agent)+'['+['x', 'y', 'z'][j]+']$'
        plt.plot(timeSeq, force[j], label = label)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    if 'fLoc' in kwargs:
      plt.gcf().set_size_inches(10*plt.gcf().get_size_inches())
      plt.tight_layout()
      plt.savefig(kwargs['fLoc']+'chaser_force.png')
      plt.close()
    else:
      plt.show()

### Matplotlib-based Plotter
def plotter(t,      x = [],      u = [],      y = [],      c =   [],
               labelX = [], labelU = [], labelY = [], labelC = None,
               *args, **kwargs):
  cmap = plt.cm.get_cmap(plt.cm.viridis, 256)
  
  if len(y) and not len(c):
    figstyle = "Y0"
  elif not len(y) and (c):
    figstyle = "0C"
  elif len(y) and len(c):
    figstyle = "YC"
  elif not len(y) and not len(c):
    figstyle = "00"

  plt.figure(figsize=(4,3))
  for i in range(len(x)):
    if figstyle == "00":
      plt.subplot(2,len(x),i+1)
    else:
      plt.subplot(3,len(x),i+1)
    if len(labelX):
      label = labelX[i]
    else:
      label = '$x_'+str(i)+'$'
    plt.plot(t, x[i], c     = cmap(80*i),
                      label = label)
    if "x1" in kwargs and i < len(kwargs["x1"]):
      if "labelX1" in kwargs:
        label = kwargs["labelX1"][i]
      else:
        label = '$x_{1,'+str(i)+'}$'
      plt.plot(t, kwargs["x1"][i], c = cmap(80*i+40),
               label = label, linestyle = "--")
    if "x2" in kwargs and i < len(kwargs["x2"]):
      if "labelX2" in kwargs:
        label = kwargs["labelX2"][i]
      else:
        label = '$x_{2,'+str(i)+'}$'
      plt.plot(t, kwargs["x2"][i], c = cmap(80*i+80),
               label = label, linestyle = "-.")
    plt.legend()
    if i == 0:
      plt.ylabel('State')
      plt.xlabel('Time')

  if figstyle == "00":
    plt.subplot(2,1,2)
  else:
    plt.subplot(3,1,2)
  for i in range(len(u)):
    if len(labelU):
      label = labelU[i]
    else:
      label = '$u_'+str(i)+'$'
    plt.plot(t, u[i], c     = cmap(150+80*i),
                      label = label)
  plt.legend()
  plt.ylabel('Input')
  plt.xlabel('Time')
  
  if figstyle == "Y0":
    plt.subplot(3,1,3)
    for i in range(len(y)):
      if len(labelY):
        label = labelY[i]
      else:
        label = '$y_'+str(i)+'$'
      plt.plot(t, y[i], c = cmap(180+96*i), label = label)
    plt.legend()
    plt.ylabel('Output')
    plt.xlabel('Time')
  elif figstyle == "0C":
    plt.subplot(3,1,3)
    if labelC:
      label = labelC
    else:
      label = '$c$'
    plt.plot(t, c, c = cmap(180+96*i), label = label)
    plt.legend()
    plt.ylabel('Cumulative Cost')
    plt.xlabel('Time')
  elif figstyle == "YC":
    plt.subplot(3,2,5)
    for i in range(len(y)):
      if len(labelY):
        label = labelY[i]
      else:
        label = '$y_'+str(i)+'$'
      plt.plot(t, y[i], c = cmap(180+96*i), label = label)
    plt.legend()
    plt.ylabel('Output')
    plt.xlabel('Time')
    plt.subplot(3,2,6)
    if labelC:
      label = labelC
    else:
      label = '$c$'
    plt.plot(t, c, c = cmap(180+96*i), label = label)
    plt.legend()
    plt.ylabel('Cumulative Cost')
    plt.xlabel('Time')
  
  if 'fName' not in kwargs:
    plt.show()
  else:
    plt.gcf().set_size_inches(10*plt.gcf().get_size_inches())
    plt.tight_layout()
    plt.savefig(kwargs['fName'])
    plt.close()

### Plotly-based Simulator
def simulate(fName, X, Y, Z, labels, numAgents, *args, **kwargs):
  markers = []
  for agent in range(numAgents):
    if 'lines' in kwargs and kwargs['lines'] == True:
      markers.append(go.Scatter3d(mode = 'lines',
                                  x = X[agent], y = Y[agent], z = Z[agent],
                                  marker = dict(size       = 3,),
                                  name = labels[agent]))
    if 'markers' not in kwargs or kwargs['markers'] == True:
      markers.append(go.Scatter3d(mode = 'markers',
                                  x = X[agent], y = Y[agent], z = Z[agent],
                                  marker = dict(size  = 3,                                 
                                                color = 'darkblue',),
                                  name = labels[agent]))
  layout = go.Layout(
    font_color = 'white',
    paper_bgcolor='rgba(72,72,72,255)',
    plot_bgcolor='rgba(185,185,185,255)'
  )

  if "pointList" in kwargs:
    pointList = kwargs["pointList"]
    for i, point in enumerate(pointList):
      markers.append(go.Scatter3d(mode = 'markers',
                                  x = [point[0]], y = [point[1]], z = [point[2]],
                                  marker = dict(size = 6),
                                  name = "P"+str(i+1)))

  if "shape" in kwargs:
    for shape in kwargs["shape"]:
      if   shape['type'] == 'sphere':
        radius  = shape['radius']
        center  = shape['center']
        opacity = shape['opacity']
        u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
        x = radius*np.cos(u)*np.sin(v) + center[0]
        y = radius*np.sin(u)*np.sin(v) + center[1]
        z = radius*np.cos(v)           + center[2]
        markers.append(go.Surface(x = x, y = y, z = z, opacity = opacity, showscale = False,))
      elif shape['type'] == 'ellipsoid':
        radii   = shape['radii']
        center  = shape['center']
        opacity = shape['opacity']
        u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
        x = radii[0]*np.cos(u)*np.sin(v) + center[0]
        y = radii[1]*np.sin(u)*np.sin(v) + center[1]
        z = radii[2]*np.cos(v)           + center[2]
        markers.append(go.Surface(x = x, y = y, z = z, opacity = opacity, showscale = False,))
      elif shape['type'] == 'cylinder':
        radius  = shape['radius']
        length  = shape['length']
        center  = shape['center']
        opacity = shape['opacity']
        u, v = np.mgrid[0:2 * np.pi:100j, center[2]-length/2.0:center[2]+length/2.0:20j]
        x = radius*np.cos(u) + center[0]
        y = radius*np.sin(u) + center[1]
        z = v
        markers.append(go.Surface(x = x, y = y, z = z, opacity = opacity, showscale = False,))
        if 'openTop' not in shape or shape['openTop'] == False:
          u = np.linspace(0, 2*np.pi, 100)
          x, y = [], []
          for r in np.linspace(0, radius, 100):
            x.extend(r*np.cos(u) + center[0])
            y.extend(r*np.sin(u) + center[1])
          x = np.array([x for i in range(5)])
          y = np.array([y for i in range(5)])
          x = np.transpose(x)
          y = np.transpose(y)
          z = [[z_i for z_i in np.linspace(center[2]+length/2.0-0.005, center[2]+length/2.0+0.005, 3)]]*len(x)
          markers.append(go.Surface(x = x, y = y, z = z, opacity = opacity, showscale = False,))
        if 'openBottom' not in shape or shape['openBottom'] == False:
          u = np.linspace(0, 2*np.pi, 100)
          x, y = [], []
          for r in np.linspace(0, radius, 100):
            x.extend(r*np.cos(u) + center[0])
            y.extend(r*np.sin(u) + center[1])
          x = np.array([x for i in range(5)])
          y = np.array([y for i in range(5)])
          x = np.transpose(x)
          y = np.transpose(y)
          z = [[z_i for z_i in np.linspace(center[2]-length/2.0-0.005, center[2]-length/2.0+0.005, 3)]]*len(x)
          markers.append(go.Surface(x = x, y = y, z = z, opacity = opacity, showscale = False,))
  
  fig = go.Figure(data = markers, layout = layout)
  ptyplt.plot(fig, filename = fName, auto_open = False)

def orbitGenerator(constants, t, numAgents):
  mu, r_0, n, T = constants
  d_s = 10
  
  rho_x = [(agent*(d_s/2.0-0)/numAgents) for agent in range(1, numAgents+1)]                        ### 0 to d_s/2.0
  rho_y = 0
  rho_z = [agent*(d_s-0)/numAgents for agent in range(numAgents)]                                   ### 0 to 2*d_s
  alpha_x = 0
  alpha_z = [(agent*((np.pi/2)-(-np.pi/2))/numAgents - np.pi/2) for agent in range(numAgents)]      ### -np.pi/2 to np.pi/2
  
  random.shuffle(rho_x)
  random.shuffle(rho_z)
  random.shuffle(alpha_z)
  
  X = [[rho_x[agent]*np.sin(n*t[elem]+alpha_x)             for elem in range(len(t))] for agent in range(numAgents)]
  Y = [[rho_y + 2.0*rho_x[agent]*np.cos(n*t[elem]+alpha_x) for elem in range(len(t))] for agent in range(numAgents)]
  Z = [[rho_z[agent]*np.sin(n*t[elem]+alpha_z[agent])      for elem in range(len(t))] for agent in range(numAgents)]
  
  return X, Y, Z

def orbitImporter():
  fName  = latestDataFile('../results/')
  data   = loadData(fName)
  ipData = data["ipData"]
  opData = data["opData"]
  x, u, y, d = opData
  return [x[0]], [x[1]], [x[2]], fName

# MAIN PROGRAM
if __name__ == '__main__':
  mu  = 3.986004418*(10**14)                   ## standard gravitational parameter 
  r_0 = (6371+400)*(10**3)                     ## radius of the chief orbit in ECI frame
  n   = np.sqrt(mu/r_0**3)                     ## mean motion of the target spacecraft
  T   = 2*np.pi/n                              ## time period of orbit
  
  constants = (mu, r_0, n, T)
  timeSeq   = [time.item() for time in np.arange(0,T,T/1000)]
  
  remote = True
  if remote == False:
    numAgents = 5
    X, Y, Z   = orbitGenerator(constants, timeSeq, numAgents)
    fName     = filenameCreator('../plots/', '.html')
    simulate(fName, X, Y, Z, [(agent+1) for agent in range(numAgents)], numAgents)
  else:
    numAgents = 1
    X, Y, Z, fName = orbitImporter()
    fName     = '../plots/' + fName[10:-5] + '.html'
    simulate(fName, X, Y, Z, [(agent+1) for agent in range(numAgents)], numAgents)