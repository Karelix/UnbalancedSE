import autograd.numpy as np
from autograd import jacobian
import numpy as nu
from flows import *
from network import *

def printA(a):
  for row in a:
    for col in row:
      print("{:8.3f}".format(col), end=" ")
    print("")

def init_states(buses,x):
  t = x
  descs = []
  for bus in buses:
    if not bus.slack:
      t = np.concatenate((t,bus.v_states,bus.delta_states),axis=0)
    descs = descs + bus.desc
  if len(descs) == 0:
    return t
  else:
    return init_states(descs,t)

def back_equations(x,ref,y,flag):
  eqs = np.empty((0,1))
  Vi,deltai = ref[0:3], ref[3:6]
  Vj,deltaj = x[0:3], x[3:6]
  if flag:
    p = Pflow(Vi,deltai,Vj,deltaj,y,'ij')
    q = Qflow(Vi,deltai,Vj,deltaj,y,'ij')
    eqs = np.concatenate((eqs,p,q),axis=0)
  p = -Pflow(Vi,deltai,Vj,deltaj,y,'ji')
  q = -Qflow(Vi,deltai,Vj,deltaj,y,'ji')
  eqs = np.concatenate((eqs,p,q),axis=0)
  return eqs

def bse(net,anc,curr,jac):
  '''
  bse looks recursively every level of the network bottom to top and
  examines every bus using as reference its ancestor's state, the meaurements
  of the line between them and the cumulative power injection provided form 
  previous iterations of bse    
  '''
  # Calculate cumulative power injection on current bus
  s,sigmaP,sigmaQ = np.zeros((3,1)), np.ones((3,1))*0.000000001**2, np.ones((3,1))*0.000000001**2
  for d in curr.desc:
    ts,tsigmaP,tsigmaQ = bse(net,curr,d,jac)
    s = s + ts
    sigmaP = sigmaP + tsigmaP
    sigmaQ = sigmaQ + tsigmaQ

  if anc is None:
    return None,None,None
  else:
    # Lines are stored in the network using as index the tuple of the two buses it connects
    line = net.lines[(anc,curr)]

    # Collect all measurements provided for line and current bus and their varainces
    meas = np.empty((0,1))
    sigmas = np.empty((0,1))
    flag = False
    # 'ij' means that it is the power flow from ancestor to current bus
    if 'ij' in line.meas:
      meas = np.concatenate((meas,line.meas['ij']['p'].m,line.meas['ij']['q'].m),axis=0)
      sigmas = np.concatenate((sigmas,line.meas['ij']['p'].sigma**2,line.meas['ij']['q'].sigma**2),axis=0)
      flag = True
    p_inj = np.real(s) + curr.p_inj.m
    q_inj = np.imag(s) + curr.q_inj.m
    meas = np.concatenate((meas,p_inj,q_inj),axis=0)
    sigmas = np.concatenate((sigmas,sigmaP,sigmaQ),axis=0)

    # Create weight diagonal matrix from variances
    W = 1/sigmas*np.identity(sigmas.shape[0])
    x = np.concatenate((curr.v_states,curr.delta_states),axis=0)
    ref = np.concatenate((anc.v_states,anc.delta_states),axis=0)
    y = line.y_mtx
    err = []
    i = 0
    e = 10
    print(curr.name)
    # Branch based state estimation
    while np.abs(e) > 0.00001 and i < 100:
      H = jac(x,ref,y,flag).reshape(-1,len(list(x)))
      G = H.T.dot(W).dot(H)
      b = H.T.dot(W).dot(meas-back_equations(x,ref,y,flag))
      dx = nu.linalg.solve(G,b)
      e = np.sum(dx)/len(dx)
      # print(e)
      err.append(e)
      i = i + 1
      x = x + dx

    curr.v_states = x[0:3]
    curr.delta_states = x[3:6]
    p = Pflow(anc.v_states,anc.delta_states,curr.v_states,curr.delta_states,y,'ij')
    q = Qflow(anc.v_states,anc.delta_states,curr.v_states,curr.delta_states,y,'ij')
    new_sigmaP,new_sigmaQ = sigmaP,sigmaQ 
    if flag:
      measP = line.meas['ij']['p'].m
      measQ = line.meas['ij']['q'].m
      new_sigmaP = np.min((line.meas['ij']['p'].sigma**2,(p-measP)**2),axis=0)
      new_sigmaQ = np.min((line.meas['ij']['q'].sigma**2,(q-measQ)**2),axis=0)
    line.p_eq = Measurement(t='PFij', m=p, sigma=np.sqrt(new_sigmaP))
    line.q_eq = Measurement(t='QFij', m=q, sigma=np.sqrt(new_sigmaQ))
    return p+1j*q, new_sigmaP, new_sigmaQ

def print_states(buses):
  '''
  Recursively print the network states in breadth first search order
  '''
  descs = []
  for bus in buses:
    if not bus.slack:
      print(bus.name)
      for i in range(3):
        print(str(bus.v_states[i,0]*bus.v_base)+'/'+str(np.degrees(bus.delta_states[i,0])))
    descs = descs + bus.desc
  if len(descs) == 0:
    return 
  else:
   print_states(descs)

def backward(net):
  '''
  Backward sweep for state estimation algorithm with bfs
  '''

  # jac is a function that creates the jacobian matrix of back_equations based on the x vector provided
  jac = jacobian(back_equations)
  _,_,_ = bse(net,None,net.buses[1],jac)
  # print_states([net.buses[1]])

def fwd_equations(x,ref,y):
  '''
  Equations used in forward sweep are simply the 
  power flows from reference bus to current bus.
  '''
  Vi,deltai = ref[0:3], ref[3:6]
  Vj,deltaj = x[0:3], x[3:6]
  p = Pflow(Vi,deltai,Vj,deltaj,y,'ij')
  q = Qflow(Vi,deltai,Vj,deltaj,y,'ij')
  eqs = np.concatenate((p,q),axis=0)
  return eqs

def fse(net,anc,curr,jac):
  '''
  fse looks, recursively, the whole network using depth first search.
  For every bus, it examines the line between this bus and its ancestor bus.
  Using the power flow of this line, calculated in the backward sweep, and 
  the states of the ancestor bus, calculated in a previous iteration of the 
  fse, as reference, fse claculates the definite states of the bus for this 
  iteration of backward/forward sweep.
  '''
  if anc is not None:
    line = net.lines[(anc,curr)]
    meas = line.p_eq.m
    meas = np.concatenate((meas,line.q_eq.m),axis=0)
    sigma = line.p_eq.sigma**2
    sigma = np.concatenate((sigma,line.q_eq.sigma**2),axis=0)
    W = 1/sigma*np.identity(sigma.shape[0])
    x = np.concatenate((curr.v_states,curr.delta_states),axis=0)
    ref = np.concatenate((anc.v_states,anc.delta_states),axis=0)
    y = line.y_mtx
    err = []
    i = 0
    e = 10
    print(curr.name)
    while np.abs(e) > 0.00001 and i < 100:
      H = jac(x,ref,y).reshape(-1,len(list(x)))
      G = H.T.dot(W).dot(H)
      b = H.T.dot(W).dot(meas-fwd_equations(x,ref,y))
      dx = nu.linalg.solve(G,b)
      e = np.sum(dx)/len(dx)
      # print(e)
      err.append(e)
      i = i + 1
      x = x + dx

    curr.v_states = x[0:3]
    curr.delta_states = x[3:6]
  
  for d in curr.desc:
    fse(net,curr,d,jac)

def forward(net):
  '''
  Forward sweep for state estimation algorithm with bfs
  '''
  # jac is a function that creates the jacobian matrix of fwd_equations based on the x vector provided
  jac = jacobian(fwd_equations)
  fse(net,None,net.buses[1],jac)

def state_estimation(net):
  
  # Putting all states in a vector to monitor the error
  # States are initialized using the initial state from the buses(flat voltage profile)
  # init_states simply looks for the states stored in network's bus objects and puts them in bfs order in a vector
  x = init_states([net.buses[1]],np.empty((0,1)))

  i = 0
  err = []
  e = 10
  while e > 0.00001 and i < 100:
    print('Backward',i)
    backward(net)
    print('Forward',i)
    forward(net)
    # After forward, the updated states are stored in the bus objects
    # init_states simply looks for the states stored in network's bus objects and puts them in bfs order in a vector
    new_x = init_states([net.buses[1]],np.empty((0,1)))
    # dx is the abdolute difference between the old and the new states
    dx = np.abs(new_x - x)
    # calculate the mean of dx
    e = sum(dx)/len(dx)
    err = err + [e]
    x = new_x
    i = i + 1
  print_states([net.buses[1]])

if __name__ == '__main__':

  net = Network()

  s_base = 6000000
  zy = np.array([
                  [0.4576+1.078j,0.1559+0.5017j,0.1535+0.3849j],
                  [0.1559+0.5017j,0.4666+1.0482j,0.158+0.4236j],
                  [0.1535+0.3849j,0.158+0.4236j,0.4615+1.0651j]
                ])
  tr = (1+6j)/100

  v_base = 12470
  z_base = v_base**2/s_base
  i_base = s_base/(np.sqrt(3)*v_base)

  net.add_bus(Bus(1,slack=True,name='Bus 1',v_base=v_base/np.sqrt(3),config='4'),1)
  slack_idx = 1

  # Bus 2 IEEE 4-bus feeder
  net.add_bus(Bus(2,name='Bus 2',v_base=v_base/np.sqrt(3),config='4'),2)
  
  v_base = 4160
  # Bus 3 IEEE 4-bus feeder
  net.add_bus(Bus(3,name='Bus 3',v_base=v_base/np.sqrt(3),config='4'),3)

  # Bus 4 IEEE 4-bus feeder Unbalanced
  s = np.array([[1275000/0.85,1800000/0.9,2375000/0.95]]).T*np.exp(1j*np.arccos([[0.85,0.9,0.95]])).T/(s_base/3)
  p = np.real(s)
  q = np.imag(s)
  net.add_bus(Bus(4,p_inj=p,q_inj=q,p_sigma=0.1,q_sigma=0.001,name='Bus 4',v_base=v_base/np.sqrt(3),config='4'),4)

  # Sort buses by index. Slack always has index 1
  net.buses = {k:v for k,v in sorted(net.buses.items())}

  v_base = 12470
  z_base = v_base**2/s_base
  i_base = s_base/(np.sqrt(3)*v_base)

  b = (net.buses[1],net.buses[2])
  l = Line(idx=0,buses=b,z_mtx=zy*f_to_m(2000)/z_base,i_base=i_base,name='Line 1-2',config='4-wire') # 2000ft = 0.3787879 miles
  net.add_line(l,b)

  V1 = 12470/np.sqrt(3)*np.exp(1j*np.array([[0],[-2*np.pi/3],[2*np.pi/3]]))/(v_base/np.sqrt(3))
  # Power Flow from Bus 1 to Bus 2
  V2 = np.array([[7164,7110,7082]]).T*np.exp(1j*np.deg2rad([[-0.1,-120.2,119.3]])).T/(v_base/np.sqrt(3))
  
  P = Pflow(np.abs(V1),np.angle(V1),np.abs(V2),np.angle(V2),l.y_mtx,'ij')
  Q = Qflow(np.abs(V1),np.angle(V1),np.abs(V2),np.angle(V2),l.y_mtx,'ij')
  l.meas['ij'] = {}
  l.meas['ij']['p'] = Measurement(t='PFij',m=P,sigma=0.2*np.ones((3,1)))
  l.meas['ij']['q'] = Measurement(t='QFij',m=Q,sigma=0.2*np.ones((3,1)))

  v_base = 4160
  z_base = v_base**2/s_base
  i_base = s_base/(np.sqrt(3)*v_base)
  
  b = (net.buses[2],net.buses[3])
  l = Line(idx=1,buses=b,z_mtx=None,i_base=None,trans=True,trans_conf='YgYg',step='down',trans_z=tr,name='Transformer')
  net.add_line(l,b)

  V3 = np.array([[2305,2255,2203]]).T*np.exp(1j*np.deg2rad([[-2.3,-123.6,114.8]])).T/(v_base/np.sqrt(3))
  P = Pflow(np.abs(V2),np.angle(V2),np.abs(V3),np.angle(V3),l.y_mtx,'ij')
  Q = Qflow(np.abs(V2),np.angle(V2),np.abs(V3),np.angle(V3),l.y_mtx,'ij')
  l.meas['ij'] = {}
  l.meas['ij']['p'] = Measurement(t='PFij',m=P,sigma=0.2*np.ones((3,1)))
  l.meas['ij']['q'] = Measurement(t='QFij',m=Q,sigma=0.2*np.ones((3,1)))

  b = (net.buses[3],net.buses[4])
  l = Line(idx=2,buses=b,z_mtx=zy*f_to_m(2500)/z_base,i_base=i_base,name='Line 3-4',config='4-wire') # 2500ft = 0.4734848 miles
  net.add_line(l,b)

  V4 = np.array([[2175,1930,1833]]).T*np.exp(1j*np.deg2rad([[-4.1,-126.8,102.8]])).T/(v_base/np.sqrt(3))
  # Power Flow from Bus 3 to Bus 4
  P = Pflow(np.abs(V3),np.angle(V3),np.abs(V4),np.angle(V4),l.y_mtx,'ij')
  Q = Qflow(np.abs(V3),np.angle(V3),np.abs(V4),np.angle(V4),l.y_mtx,'ij')
  l.meas['ij'] = {}
  l.meas['ij']['p'] = Measurement(t='PFij',m=P,sigma=0.2*np.ones((3,1)))
  l.meas['ij']['q'] = Measurement(t='QFij',m=Q,sigma=0.2*np.ones((3,1)))
  
  # create_levels(net)
  state_estimation(net)