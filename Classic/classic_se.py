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

def init_states(x,buses,meas,sigma,pos):
  # t = x
  # m = meas
  # s = sigma
  descs = []
  for anc,bus in buses:
    if not bus.slack:
      x = np.concatenate((x,bus.v_states,bus.delta_states),axis=0)
      for msmt in net.lines[(anc,bus)].meas:
        meas = np.concatenate((meas,msmt.m),axis=0)
        sigma = np.concatenate((sigma,msmt.sigma),axis=0)
      for msmt in bus.meas:
        meas = np.concatenate((meas,msmt.m),axis=0)
        sigma = np.concatenate((sigma,msmt.sigma),axis=0)
      for i in range(3):
        bus.v_pos[i] = pos+i
        bus.delta_pos[i] = pos+i+3
      pos = pos+6
    else:
      x = np.concatenate((x,bus.v_states),axis=0)
      for msmt in bus.meas:
        meas = np.concatenate((meas,msmt.m),axis=0)
        sigma = np.concatenate((sigma,msmt.sigma),axis=0)
      for i in range(3):
        bus.v_pos[i] = pos+i
      pos = pos+3
    descs = descs + [(bus,d) for d in bus.desc]
  if len(descs) == 0:
    return x,meas,sigma
  else:
    return init_states(x,descs,meas,sigma,pos)

def equations(x,buses,net,eqs):
  descs = []
  # x = x.reshape((-1,1),order='C')
  for anc,bus in buses:
    if not bus.slack:
      anc_v = x[anc.v_pos[0]:anc.v_pos[0]+3]
      anc_delta = x[anc.delta_pos[0]:anc.delta_pos[0]+3] if not anc.slack else anc.delta_states
      bus_v = x[bus.v_pos[0]:bus.v_pos[0]+3]
      bus_delta = x[bus.delta_pos[0]:bus.delta_pos[0]+3]
      for msmt in net.lines[(anc,bus)].meas:
        if msmt.t[0:2] == 'PF':
          eq = Pflow(anc_v,anc_delta,bus_v,bus_delta,net.lines[(anc,bus)].y_mtx,msmt.t[2:])
        elif msmt.t[0:2] == 'QF':
          eq = Qflow(anc_v,anc_delta,bus_v,bus_delta,net.lines[(anc,bus)].y_mtx,msmt.t[2:])
        eqs = np.concatenate((eqs,eq),axis=0)
      for msmt in bus.meas:
        if msmt.t == 'V':
          eq = bus_v
        elif msmt.t == 'PI':
          eq = -Pflow(anc_v,anc_delta,bus_v,bus_delta,net.lines[(anc,bus)].y_mtx,'ji')
          for d in bus.desc:
            d_v = x[d.v_pos[0]:d.v_pos[0]+3]
            d_delta = x[d.delta_pos[0]:d.delta_pos[0]+3]
            eq = eq + Pflow(bus_v,bus_delta,d_v,d_delta,net.lines[(bus,d)].y_mtx,'ij')
        elif msmt.t == 'QI':
          eq = -Qflow(anc_v,anc_delta,bus_v,bus_delta,net.lines[(anc,bus)].y_mtx,'ji')
          for d in bus.desc:
            d_v = x[d.v_pos[0]:d.v_pos[0]+3]
            d_delta = x[d.delta_pos[0]:d.delta_pos[0]+3]
            eq = eq + Qflow(bus_v,bus_delta,d_v,d_delta,net.lines[(bus,d)].y_mtx,'ij')
        eqs = np.concatenate((eqs,eq),axis=0)
    else:
      for msmt in bus.meas:
        if msmt.t == 'V':
          eq = x[bus.v_pos[0]:bus.v_pos[0]+3]
        eqs = np.concatenate((eqs,eq),axis=0)  
    descs = descs + [(bus,d) for d in bus.desc]
  if len(descs) == 0:
    return eqs
  else:
    # x = x.flatten()
    return equations(x,descs,net,eqs)

def print_phases(v,angles):
  for i in range(3):
    print(str(v[i,0])+'/'+str(np.degrees(angles[i,0])))

def print_states(x,buses):
  descs = []
  for bus in buses:
    if not bus.slack:
      bus_v = x[bus.v_pos[0]:bus.v_pos[0]+3]*bus.v_base
      bus_delta = x[bus.delta_pos[0]:bus.delta_pos[0]+3]
      print(bus.name)
      print_phases(bus_v,bus_delta)
    descs = descs + bus.desc
  if not len(descs) == 0:
    print_states(x,descs)

def state_estimation(net):
  x,meas,sigma = init_states(np.empty((0,1)),[(None,net.buses[1])],np.empty((0,1)),np.empty((0,1)),0)
  # x = x.flatten()
  # print(meas*6000000/3)
  # input()
  # print(equations(x,[(None,net.buses[1])],net,np.empty((0,1))))
  # input()
  jac = jacobian(equations)

  W = sigma**(-2)*np.identity(sigma.shape[0])
  err = []
  i = 0
  e = 10
  # Branch based state estimation
  while np.abs(e) > 0.00001 and i < 100:
    print(i)
    H = jac(x,[(None,net.buses[1])],net,np.empty((0,1))).reshape(-1,len(list(x)))
    # H = jac(x,[(None,net.buses[1])],net,np.empty((0,1)))
    # printA(H)
    # input()
    G = H.T.dot(W).dot(H)
    b = H.T.dot(W).dot(meas-equations(x,[(None,net.buses[1])],net,np.empty((0,1))))
    dx = nu.linalg.solve(G,b)
    e = np.sum(dx)/len(dx)
    print(e)
    err.append(e)
    i = i + 1
    x = x + dx
  
  print_states(x.reshape((-1,1),order='F'),[net.buses[1]])

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
  v = np.ones((3,1))
  net.buses[1].add_measurement(Measurement('V',m=v,sigma=0.001*np.ones((3,1))))
  slack_idx = 1

  # Bus 2 IEEE 4-bus feeder
  net.add_bus(Bus(2,name='Bus 2',v_base=v_base/np.sqrt(3),config='4'),2)
  # p = np.zeros((3,1))
  # q = np.zeros((3,1))
  # net.buses[2].add_measurement(Measurement('PI',m=p,sigma=0.02*np.ones((3,1))))
  # net.buses[2].add_measurement(Measurement('QI',m=q,sigma=0.02*np.ones((3,1))))

  v_base = 4160
  # Bus 3 IEEE 4-bus feeder
  net.add_bus(Bus(3,name='Bus 3',v_base=v_base/np.sqrt(3),config='4'),3)
  # p = np.zeros((3,1))
  # q = np.zeros((3,1))
  # net.buses[3].add_measurement(Measurement('PI',m=p,sigma=0.02*np.ones((3,1))))
  # net.buses[3].add_measurement(Measurement('QI',m=q,sigma=0.02*np.ones((3,1))))

  # Bus 4 IEEE 4-bus feeder Unbalanced
  s = np.array([[1275000/0.85,1800000/0.9,2375000/0.95]]).T*np.exp(1j*np.arccos([[0.85,0.9,0.95]])).T/(s_base/3)
  p = np.real(s)
  q = np.imag(s)
  net.add_bus(Bus(4,name='Bus 4',v_base=v_base/np.sqrt(3),config='4'),4)
  net.buses[4].add_measurement(Measurement('PI',m=p,sigma=0.001*np.ones((3,1))))
  net.buses[4].add_measurement(Measurement('QI',m=q,sigma=0.001*np.ones((3,1))))


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
  l.add_measurement(Measurement(t='PFij',m=P,sigma=0.2*np.ones((3,1))))
  l.add_measurement(Measurement(t='QFij',m=Q,sigma=0.2*np.ones((3,1))))

  v_base = 4160
  z_base = v_base**2/s_base
  i_base = s_base/(np.sqrt(3)*v_base)
  
  b = (net.buses[2],net.buses[3])
  l = Line(idx=1,buses=b,z_mtx=None,i_base=None,trans=True,trans_conf='YgYg',step='down',trans_z=tr,name='Transformer')
  net.add_line(l,b)

  V3 = np.array([[2305,2255,2203]]).T*np.exp(1j*np.deg2rad([[-2.3,-123.6,114.8]])).T/(v_base/np.sqrt(3))
  P = Pflow(np.abs(V2),np.angle(V2),np.abs(V3),np.angle(V3),l.y_mtx,'ij')
  Q = Qflow(np.abs(V2),np.angle(V2),np.abs(V3),np.angle(V3),l.y_mtx,'ij')
  l.add_measurement(Measurement(t='PFij',m=P,sigma=0.2*np.ones((3,1))))
  l.add_measurement(Measurement(t='QFij',m=Q,sigma=0.2*np.ones((3,1))))

  b = (net.buses[3],net.buses[4])
  l = Line(idx=2,buses=b,z_mtx=zy*f_to_m(2500)/z_base,i_base=i_base,name='Line 3-4',config='4-wire') # 2500ft = 0.4734848 miles
  net.add_line(l,b)

  V4 = np.array([[2175,1930,1833]]).T*np.exp(1j*np.deg2rad([[-4.1,-126.8,102.8]])).T/(v_base/np.sqrt(3))
  # Power Flow from Bus 3 to Bus 4
  P = Pflow(np.abs(V3),np.angle(V3),np.abs(V4),np.angle(V4),l.y_mtx,'ij')
  Q = Qflow(np.abs(V3),np.angle(V3),np.abs(V4),np.angle(V4),l.y_mtx,'ij')
  l.add_measurement(Measurement(t='PFij',m=P,sigma=0.2*np.ones((3,1))))
  l.add_measurement(Measurement(t='QFij',m=Q,sigma=0.2*np.ones((3,1))))
  
  # create_levels(net)
  state_estimation(net)