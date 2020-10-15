import numpy as np

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

class Load:
  '''
  Demand is a vector with three elements. Depending on the type of the load(Z, PQ or I), it contains Impedances, Powers or Currents.
  Depending on the configuration of the load(Y or D), the elements of the vector represent phases a,b,c or ab,bc,ca.
  Non existing phases for single or two phase elements are zero.
  ATTENTION!
  For D configuration, we call a load single phase, when it uses only ab, bc or ca. 
  '''
  def __init__(self, t='PQ', config='Y', demand=np.zeros((3,1)), demand_type='PQ', s_rated=None, v_rated=None):
    self.t = t
    self.config = config
    self.demand = demand
    self.demand_type = demand_type
    self.s_rated = s_rated
    self.v_rated = v_rated

class Generator:
  '''
  Generation is a vector with three elements. Depending on the type of the generator(PV, PQ or I), it contains Powers or Currents.
  Depending on the configuration of the load(Y or D), the elements of the list represent phases a,b,c or ab,bc,ca.
  Non existing phases for single or two phase elements are zero.
  ATTENTION!
  For D configuration, we call a load single phase, when it uses only ab, bc or ca. 
  '''
  def __init__(self, t='PQ', config='Y', generation=np.zeros((3,1)), v_ref=None, generation_type='PQ', s_rated=None, v_rated=None):
    self.t = t
    self.config = config
    self.generation = generation
    self.generation_type = generation_type
    self.s_rated = s_rated
    self.v_rated = v_rated
    if t == 'PV':
      self.v_ref = v_ref

class Bus:

  # Buses can take many loads and many generators. There can be only one PV generator!
  def __init__(self, idx, v_base, slack=False, loads=None, gens=None, name=None, config='4',level=None,level_idx=None):
    self.idx = idx
    self.v_base = v_base
    self.slack = slack
    self.loads = [] if loads==None else loads
    self.gens = [] if gens==None else gens
    self.name = name
    self.config = config
    self.level = level
    self.lvl_idx = level_idx
    self.desc = []

    A = np.sqrt(2/3)*np.array([
                              [1,0,1/np.sqrt(2)],
                              [-1/2,np.sqrt(3)/2,1/np.sqrt(2)],
                              [-1/2,-np.sqrt(3)/2,1/np.sqrt(2)]
                            ])
    v = A.T.dot(np.exp(1j*np.array([[0],[-2*np.pi/3],[2*np.pi/3]])))
    # States in ab0
    self.v_real = np.real(v)
    self.v_imag = np.imag(v)

  def add_load(self,load):
    self.loads.append(load)

  def add_gen(self,gen):
    self.gens.append(gen)

  def change_idx(self,new_idx):
    self.idx = new_idx

class Line:

  # All angles are given in degrees and then transformed to radians
  def __init__(self, idx, buses, z_mtx, i_base, trans=False, trans_conf=None, t_rated=None,ph_sh=0, reg=False, reg_conf=None, reg_a=None, name=None, config=None, level=None):
    self.idx = idx
    self.i_base = i_base
    self.buses = buses
    self.trans = trans
    self.trans_conf = trans_conf
    self.ph_sh = ph_sh
    self.reg=reg
    self.reg_conf=reg_conf
    self.reg_a=reg_a
    self.t_rated = t_rated
    self.level = level
    self.current = np.empty((6,1))

    t = np.array([
                  [1,-1,0],
                  [0,1,-1],
                  [-1,0,1]
                ])
    ta = np.array([
                   [0,-1,0],
                   [0,1,-1],
                   [0,0,1]
                  ])
    tb = np.array([
                   [1,0,0],
                   [0,0,-1],
                   [-1,0,1]
                  ])
    tc = np.array([
                   [1,-1,0],
                   [0,1,0],
                   [-1,0,0]
                  ])

    self.z_mtx = t.dot(z_mtx) if config=='3-wire' else z_mtx

    self.name = name

    # Based on 2013 paper "Unbalanced Power Flow in Distribution Systems With Embedded Transformers Using the Complex Theory in alpha beta 0 Stationary Reference Frame"
    self.n1 = np.identity(3)
    self.n2 = np.identity(3)
    self.n3 = -np.identity(3)
    self.n4 = np.identity(3)
    if self.trans:
      if self.trans_conf == 'YgYg':
        self.n1 = np.identity(3)
        self.n2 = trans_connection(theta=-ph_sh,rot=ph_sh)
        self.n3 = -trans_connection(theta=ph_sh,rot=ph_sh)
      elif self.trans_conf == 'DD':
        self.n1 = np.sqrt(3)*trans_connection(theta=30)
        self.n2 = trans_connection(theta=-ph_sh,rot=ph_sh)
        self.n3 = -trans_connection(theta=ph_sh,rot=ph_sh)
      elif self.trans_conf == 'YgD':
        self.n1 = np.identity(3)
        self.n2 = trans_connection(theta=-ph_sh-30,rot=ph_sh+30)/np.sqrt(3)
        self.n3 = -trans_connection(theta=ph_sh)
      elif self.trans_conf == 'YD':
        self.n1 = trans_connection(theta=30)*np.sqrt(3)
        self.n2 = trans_connection(theta=-ph_sh)
        self.n3 = -trans_connection(theta=ph_sh)
      elif self.trans_conf == 'DYg':
        self.n1 = trans_connection(theta=30-ph_sh,rot=ph_sh-30)*np.sqrt(3)
        self.n2 = trans_connection(theta=30-ph_sh,rot=ph_sh-30)*np.sqrt(3)
        self.n4 = trans_connection(theta=-ph_sh)
    elif self.reg:
      if self.reg_conf == 'YA':
        k = np.array([
                      [self.reg_a[0], 0, 0], 
                      [0, self.reg_a[1], 0], 
                      [0, 0, self.reg_a[2]]
                    ])
        l = np.array([
                      [self.reg_a[0], 0, 0], 
                      [0, self.reg_a[1], 0], 
                      [0, 0, self.reg_a[2]]
                    ])
        k_rec = np.reciprocal(k,where=(k!=0))
        self.n1 = convertNfromABCtoAB0(k_rec)
        self.n2 = convertNfromABCtoAB0(k_rec)
        self.n4 = convertNfromABCtoAB0(l)
      elif self.reg_conf == 'YB':
        k = np.array([
                      [self.reg_a[0], 0, 0], 
                      [0, self.reg_a[1], 0], 
                      [0, 0, self.reg_a[2]]
                    ])
        self.n1 = convertNfromABCtoAB0(np.identity(3))
        self.n2 = convertNfromABCtoAB0(k)
        self.n3 = convertNfromABCtoAB0(-k)
      elif self.reg_conf == 'DA':
        k = np.array([
                      [self.reg_a[0], 1-self.reg_a[1], 0], 
                      [0, self.reg_a[1], 1-self.reg_a[2]], 
                      [1-self.reg_a[0], 0, self.reg_a[2]]
                    ])
        l = np.array([
                      [self.reg_a[0], 0, 1-self.reg_a[2]], 
                      [1-self.reg_a[0], self.reg_a[1], 0], 
                      [0, 1-self.reg_a[1], self.reg_a[2]]
                    ])
        self.n1 = convertNfromABCtoAB0(np.linalg.inv(k).dot(t))
        self.n2 = convertNfromABCtoAB0(np.linalg.inv(k))
        self.n4 = convertNfromABCtoAB0(l)
      elif self.reg_conf == 'DB':
        k = np.array([
                      [self.reg_a[0], 1-self.reg_a[1], 0], 
                      [0, self.reg_a[1], 1-self.reg_a[2]], 
                      [1-self.reg_a[0], 0, self.reg_a[2]]
                    ])
        l = np.array([
                      [self.reg_a[0], 0, 1-self.reg_a[2]], 
                      [1-self.reg_a[0], self.reg_a[1], 0], 
                      [0, 1-self.reg_a[1], self.reg_a[2]]
                    ])
        self.n1 = convertNfromABCtoAB0(t)
        self.n2 = convertNfromABCtoAB0(k)
        self.n3 = convertNfromABCtoAB0(-l)
      elif self.reg_conf == 'oDAa':
        k = np.array([
                      [self.reg_a[0], 0, 0], 
                      [-self.reg_a[0], 0, -self.reg_a[2]], 
                      [0, 0, self.reg_a[2]]
                    ])
        l = np.array([
                      [0, -self.reg_a[0], -self.reg_a[2]], 
                      [0, self.reg_a[0], 0], 
                      [0, 0, self.reg_a[2]]
                    ])
        k_rec = np.reciprocal(k,where=(k!=0)) 
        self.n1 = convertNfromABCtoAB0(k_rec.dot(ta))
        self.n2 = convertNfromABCtoAB0(k_rec)
        self.n4 = convertNfromABCtoAB0(l)
      elif self.reg_conf == 'oDAb':
        k = np.array([
                      [self.reg_a[0], 0, 0], 
                      [0, self.reg_a[1], 0], 
                      [-self.reg_a[0], -self.reg_a[1], 0]
                    ])
        l = np.array([
                      [self.reg_a[0], 0, 0], 
                      [-self.reg_a[0], 0, -self.reg_a[1]], 
                      [0, 0, self.reg_a[1]]
                    ])
        k_rec = np.reciprocal(k,where=(k!=0))
        self.n1 = convertNfromABCtoAB0(k_rec.dot(tb))
        self.n2 = convertNfromABCtoAB0(k_rec)
        self.n4 = convertNfromABCtoAB0(l)
      elif self.reg_conf == 'oDAc':
        k = np.array([
                      [0, -self.reg_a[1], -self.reg_a[2]], 
                      [0, self.reg_a[1], 0], 
                      [0, 0, self.reg_a[2]]
                    ])
        l = np.array([
                      [self.reg_a[1], 0, 0], 
                      [0, self.reg_a[2], 0], 
                      [-self.reg_a[1], -self.reg_a[2], 0]
                    ])
        k_rec = np.reciprocal(k,where=(k!=0))
        self.n1 = convertNfromABCtoAB0(k_rec.dot(tc))
        self.n2 = convertNfromABCtoAB0(k_rec)
        self.n4 = convertNfromABCtoAB0(l)
      elif self.reg_conf == 'oDBa':
        k = np.array([
                      [self.reg_a[0], 0, 0], 
                      [-self.reg_a[0], 0, -self.reg_a[2]], 
                      [0, 0, self.reg_a[2]]
                    ])
        l = np.array([
                      [0, -self.reg_a[0], -self.reg_a[2]], 
                      [0, self.reg_a[0], 0], 
                      [0, 0, self.reg_a[2]]
                    ])
        l_rec = np.reciprocal(l,where=(l!=0))
        self.n1 = convertNfromABCtoAB0(ta)
        self.n2 = convertNfromABCtoAB0(k)
        self.n3 = convertNfromABCtoAB0(-l_rec)
      elif self.reg_conf == 'oDBb':
        k = np.array([
                      [self.reg_a[0], 0, 0], 
                      [0, self.reg_a[1], 0], 
                      [-self.reg_a[0], -self.reg_a[1], 0]
                    ])
        l = np.array([
                      [self.reg_a[0], 0, 0], 
                      [-self.reg_a[0], 0, -self.reg_a[1]], 
                      [0, 0, self.reg_a[1]]
                    ])
        l_rec = np.reciprocal(l,where=(l!=0))
        self.n1 = convertNfromABCtoAB0(tb)
        self.n2 = convertNfromABCtoAB0(k)
        self.n3 = convertNfromABCtoAB0(-l_rec)
      elif self.reg_conf == 'oDBc':
        k = np.array([
                      [0, -self.reg_a[1], -self.reg_a[2]], 
                      [0, self.reg_a[1], 0], 
                      [0, 0, self.reg_a[2]]
                    ])
        l = np.array([
                      [self.reg_a[2], 0, 0], 
                      [0, self.reg_a[1], 0], 
                      [-self.reg_a[2], -self.reg_a[1], 0]
                    ])
        l_rec = np.reciprocal(l,where=(l!=0))
        self.n1 = convertNfromABCtoAB0(tc)
        self.n2 = convertNfromABCtoAB0(k)
        self.n3 = convertNfromABCtoAB0(-l_rec)

  def change_idx(self,new_idx):
    self.idx = new_idx

def delta_func(rot):
  '''
  This function calculates the delta(rotation) based on the rotation given in DEGREES.
  This calculation is based on the Equation given in the TABLE I in the 2013 paper
  "Unbalanced Power Flow in Distribution Systems With Embedded Transformers Using the Complex Theory in alpha beta 0 Stationary Reference Frame"
  '''
  return (1 if (rot/120+1).is_integer() else -1)

def trans_connection(theta,rot=None):
  '''
  This function takes as input theta and rotation in DEGREES.
  Based on TABLE I from 2013 paper "Unbalanced Power Flow in Distribution Systems With Embedded Transformers Using the Complex Theory in alpha beta 0 Stationary Reference Frame"
  It calculates G for the theta given and delta(roation) for the rotation given, to produce a 3x3 matrix that represents N1,N2,N3 or N4 of a transformer connection.
  '''
  d = 0 if rot == None else delta_func(rot)
  return np.array([[np.cos(np.radians(theta)),-np.sin(np.radians(theta)),0],[np.sin(np.radians(theta)),np.cos(np.radians(theta)),0],[0,0,d]])

def bus_levels(net,b,i):
  print(b.name,i)
  net.add_bus_to_level(b,i)
  for d in b.desc:
    bus_levels(net,d,i+1)

def convertNfromABCtoAB0(ABC):
  A = np.sqrt(2/3)*np.array([[1,0,1/np.sqrt(2)],[-1/2,np.sqrt(3)/2,1/np.sqrt(2)],[-1/2,-np.sqrt(3)/2,1/np.sqrt(2)]])
  AB0 = A.T.dot(ABC.dot(A))
  return(AB0)