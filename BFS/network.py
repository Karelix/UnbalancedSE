import autograd.numpy as np

class Network:

  def __init__(self):
    self.buses = {}
    self.lines = {}
    self.bus_idx_by_name = {}
    self.line_idx_by_name = {}
    self.bus_levels = {}
    self.bus_lvl_length = {}
    self.line_levels = {}
    self.line_lvl_length = {}

  def add_bus(self,bus,idx):
    self.buses[idx] = bus
    self.bus_idx_by_name[bus.name] = idx

  def add_line(self,line,key):
    self.lines[key] = line
    key[0].desc.append(key[1])
    self.line_idx_by_name[line.name] = key

  def add_bus_to_level(self,b,i):
    b.level = i
    if i in self.bus_levels:
      j = len(self.bus_levels[i])
      b.lvl_idx = j+1
      self.bus_levels[i].append(b.name)
    else:
      b.lvl_idx = 1
      self.bus_levels[i] = [b.name]

  def add_line_to_level(self,l,i):
    l.level = i
    if i in self.line_levels:
      self.line_levels[i].append(l.name)
    else:
      self.line_levels[i] = [l.name]

  def compute_bus_lvl_length(self):
    for i in self.bus_levels:
      self.bus_lvl_length[i] = len(self.bus_levels[i])

  def compute_line_lvl_length(self):
    for i in self.line_levels:
      self.line_lvl_length[i] = len(self.line_levels[i])

class Measurement:

  def __init__(self,t,m=None,sigma=None):
    self.t = t
    self.m = np.zeros((3,1)) if m is None else m
    self.sigma = 0.000000001*np.ones((3,1)) if sigma is None else sigma

class Bus:

  # Buses can take many loads and many generators. There can be only one PV generator!
  def __init__(self, idx, v_base, slack=False, p_inj=None, p_sigma=None, q_inj=None, q_sigma=None, name=None, config='4',level=None,level_idx=None):
    self.idx = idx
    self.v_base = v_base
    self.slack = slack
    self.name = name
    self.config = config
    self.level = level
    self.lvl_idx = level_idx
    self.desc = []
    
    self.p_inj = Measurement(t='PI', m=p_inj, sigma=p_sigma)
    self.q_inj = Measurement(t='QI', m=q_inj, sigma=q_sigma)
    self.p_eq = Measurement(t='PI', m=None, sigma=None)
    self.q_eq = Measurement(t='QI', m=None, sigma=None)

    self.v_states = np.ones((3,1))
    self.delta_states = np.array([[0,-2*np.pi/3,2*np.pi/3]]).T

  def change_idx(self,new_idx):
    self.idx = new_idx

class Line:

  # All angles are given in degrees and then transformed to radians
  def __init__(self, idx, buses, z_mtx, i_base, meas_ij=None, meas_ji=None, trans=False, step=None, trans_conf=None, trans_z=None, name=None, config=None, level=None):
    self.idx = idx
    self.i_base = i_base
    self.buses = buses
    self.trans = trans
    self.trans_conf = trans_conf
    # self.ph_sh = ph_sh
    # self.reg=reg
    # self.reg_conf=reg_conf
    # self.reg_a=reg_a
    # self.t_rated = t_rated
    self.level = level
    self.name = name

    self.meas = {}
    if meas_ij is not None:
      self.meas['ij'] = meas_ij
    if meas_ji is not None:
      self.meas['ji'] = meas_ji
    self.p_eq = Measurement(t='PFij', m=None, sigma=None)
    self.q_eq = Measurement(t='QFij', m=None, sigma=None)

    if not trans:
      y = np.linalg.inv(z_mtx)
      temp = np.concatenate((y,-y),axis=1)
      self.y_mtx = np.concatenate((temp,-temp),axis=0)
    else: 
      y = 1/trans_z
      I = np.identity(3)*y
      II = 1/3*(3*np.identity(3)-1)*y
      temp = np.array([[-1,1,0],
                       [0,-1,1],
                       [1,0,-1]])
      III = 1/np.sqrt(3)*temp*y
      III = III if step == 'down' else III.T

      A = I if trans_conf == 'YgYg' or trans_conf == 'YgD' else II
      B = I if trans_conf == 'YgYg' or trans_conf == 'DYg' else II
      C = -I if trans_conf == 'YgYg' else (-II if trans_conf in ['YgY','YYg','YY','DD'] else III)
      temp = np.concatenate((A,C),axis=1)
      temp2 = np.concatenate((C.T,B),axis=1)
      self.y_mtx = np.concatenate((temp,temp2),axis=0)

def bus_levels(net,b,i):
  print(b.name,i)
  net.add_bus_to_level(b,i)
  for d in b.desc:
    bus_levels(net,d,i+1)

def create_levels(net):
  bus_levels(net,net.buses[1],0)

  for b in net.lines:
    print(b[0].name,b[1].name)
    net.add_line_to_level(net.lines[b],b[0].level)
  
  net.bus_levels = {k:v for k,v in sorted(net.bus_levels.items())}
  net.line_levels = {k:v for k,v in sorted(net.line_levels.items())}

def f_to_m(feet):
  miles=feet*.00018939394
  return miles