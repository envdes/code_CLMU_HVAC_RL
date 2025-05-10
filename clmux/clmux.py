import numpy as np
from typing import List, Tuple, Dict, Any, Union

# ref: https://github.com/ESCOMP/CTSM/blob/master/src/main/clm_varcon.F90
class bem_var_con:
    
    """
    The class bem_var_con contains the variables and constants used in the Building Energy Model (BEM) for a single room.
    
    """
    
    def __init__(self) -> None:
        
        # Defining constants in Python
        self.ht_wasteheat_factor = 0.2          # wasteheat factor for urban heating (-)
        self.ac_wasteheat_factor = 0.6          # wasteheat factor for urban air conditioning (-)
        self.em_roof_int = 0.9                  # emissivity of interior surface of roof (Bueno et al. 2012, GMD)
        self.em_sunw_int = 0.9                  # emissivity of interior surface of sunwall (Bueno et al. 2012, GMD)
        self.em_shdw_int = 0.9                  # emissivity of interior surface of shadewall (Bueno et al. 2012, GMD)
        self.em_floor_int = 0.9                 # emissivity of interior surface of floor (Bueno et al. 2012, GMD)
        self.hcv_roof = 0.948                   # interior convective heat transfer coefficient for roof (W m-2 K-1)
        self.hcv_roof_enhanced = 4.040          # enhanced interior convective heat transfer coefficient for roof (W m-2 K-1)
        self.hcv_floor = 0.948                  # interior convective heat transfer coefficient for floor (W m-2 K-1)
        self.hcv_floor_enhanced = 4.040         # enhanced interior convective heat transfer coefficient for floor (W m-2 K-1)
        self.hcv_sunw = 3.076                   # interior convective heat transfer coefficient for sunwall (W m-2 K-1)
        self.hcv_shdw = 3.076                   # interior convective heat transfer coefficient for shadewall (W m-2 K-1)
        self.dz_floor = 0.1                     # floor thickness - concrete (m)
        self.dens_floor = 2.35e3                # density of floor - concrete (kg m-3)
        self.sh_floor = 880.0                   # specific heat of floor - concrete (J kg-1 K-1)
        self.cp_floor = self.dens_floor * self.sh_floor   # volumetric heat capacity of floor - concrete (J m-3 K-1)
        self.vent_ach = 0.3                     # ventilation rate (air exchanges per hour)
        self.wasteheat_limit = 100.0            # limit on wasteheat (W/m2)

class clm_var_con:
    
    """
    The class clm_var_con contains the variables and constants used in the Community Land Model (CLM).
    """
    def __init__(self) -> None:
        
        SHR_CONST_BOLTZ   = 1.38065e-23  # Boltzmann's constant ~ J/K/molecule
        SHR_CONST_AVOGAD  = 6.02214e26   # Avogadro's number ~ molecules/kmole
        SHR_CONST_RGAS    = SHR_CONST_AVOGAD*SHR_CONST_BOLTZ       # Universal gas constant ~ J/K/kmole
        SHR_CONST_MWDAIR  = 28.966       # molecular weight dry air ~ kg/kmole
        SHR_CONST_RDAIR   = SHR_CONST_RGAS/SHR_CONST_MWDAIR        # Dry air gas constant     ~ J/K/kg
        SHR_CONST_PSTD    = 101325.0     #standard pressure ~ pascals
        SHR_CONST_STEBOL  = 5.67e-8      # Stefan-Boltzmann constant ~ W/m^2/K^4
        SHR_CONST_CPDAIR  = 1.00464e3    # specific heat of dry air   ~ J/kg/K
        
        self.rair = SHR_CONST_RDAIR
        self.pstd = SHR_CONST_PSTD
        self.sb = SHR_CONST_STEBOL
        self.cpair  = SHR_CONST_CPDAIR     

class bem:
    
    def __init__(self) -> None:
        
        self.bem_var_con = bem_var_con()
        self.clm_var_con = clm_var_con()

    def bem(self,
            dtime : int,
            urban_hac : str,
            urban_explicit_ac : str,
            p_ac : float,
            ht_roof : float,
            t_building_max : float,
            t_building_min : float,
            vent_ach : float,
            canyon_hwr : float,
            wtlunit_roof : float,
            zi_roof : float,
            z_roof : float,
            tssbef_roof : float,
            t_soisno_roof : float,
            tk_roof : float,
            zi_sunw : float,
            z_sunw : float,
            tssbef_sunw : float,
            t_soisno_sunw : float,
            tk_sunw : float,
            zi_shdw : float,
            z_shdw : float,
            tssbef_shdw : float,
            t_soisno_shdw : float,
            tk_shdw : float,
            t_roof_inner_bef : float,
            t_sunw_inner_bef : float,
            t_shdw_inner_bef : float,
            t_floor_bef : float,
            t_building_bef : float,
            taf : float) -> Tuple[float, float, float, float, float, float, float, float]:

        """
        The function bem calculates the Building Energy Model (BEM) for a single room.
        
        Args:
        
        dtime : int, time step (s)
        urban_hac : str, urban heating, ventilation, and air conditioning (HAC) scheme
        urban_explicit_ac : str, urban explicit air conditioning (AC) scheme
        p_ac : float, AC adoption rate
        ht_roof : float, height of the roof (m)
        t_building_max : float, maximum building temperature (K)
        t_building_min : float, minimum building temperature (K)
        vent_ach : float, ventilation rate (air exchanges per hour)
        canyon_hwr : float, building height to width ratio
        wtlunit_roof : float, wall-to-roof ratio
        zi_roof : float, interface depth of nlevurb roof (m)
        z_roof : float, node depth of nlevurb roof (m)
        tssbef_roof : float, temperature at previous time step (K)
        t_soisno_roof : float, temperature of soil under roof (K)
        tk_roof : float, roof thermal conductivity at nlevurb interface depth (W m-1 K-1)
        zi_sunw : float, interface depth of nlevurb sunwall (m)
        z_sunw : float, node depth of nlevurb sunwall (m)
        tssbef_sunw : float, temperature at previous time step (K)
        t_soisno_sunw : float, temperature of soil under sunwall (K)
        tk_sunw : float, sunwall thermal conductivity at nlevurb interface depth (W m-1 K-1)
        zi_shdw : float, interface depth of nlevurb shadewall (m)
        z_shdw : float, node depth of nlevurb shadewall (m)
        tssbef_shdw : float, temperature at previous time step (K)
        t_soisno_shdw : float, temperature of soil under shadewall (K)
        tk_shdw : float, shadewall thermal conductivity at nlevurb interface depth (W m-1 K-1)
        t_roof_inner_bef : float, roof inner temperature at previous time step (K)
        t_sunw_inner_bef : float, sunwall inner temperature at previous time step (K)
        t_shdw_inner_bef : float, shadewall inner temperature at previous time step (K)
        t_floor_bef : float, floor temperature at previous time step (K)
        t_building_bef : float, building temperature at previous time step (K)
        taf : float, air temperature (K)
        
        """


        # ROOF
        zi_roof_innerl = zi_roof
        z_roof_innerl = z_roof
        t_roof_innerl_bef = tssbef_roof
        t_roof_innerl = t_soisno_roof
        tk_roof_innerl = tk_roof

        # SUNWALL
        zi_sunw_innerl = zi_sunw
        z_sunw_innerl = z_sunw
        t_sunw_innerl_bef = tssbef_sunw
        t_sunw_innerl = t_soisno_sunw
        tk_sunw_innerl = tk_sunw

        # SHADEWALL
        zi_shdw_innerl = zi_shdw
        z_shdw_innerl = z_shdw
        t_shdw_innerl_bef = tssbef_shdw
        t_shdw_innerl = t_soisno_shdw
        tk_shdw_innerl = tk_shdw

        # Initialization of constants and variables
        bem_var_con = self.bem_var_con

        #* CONSTANTS
        #* 1 heat transfer coefficients
        # if roof temperature is greater than building temperature,
        # use normal convective heat transfer coefficient
        # else use enhanced convective heat transfer coefficient
        
        if t_roof_inner_bef <= t_building_bef:
            hcv_roofi = bem_var_con.hcv_roof_enhanced
        else:
            hcv_roofi = bem_var_con.hcv_roof
        
        #hcv_roofi = bem_var_con.hcv_roof_enhanced if t_roof_inner_bef <= t_building_bef else bem_var_con.hcv_roof

        # if floor temperature is less than building temperature,
        # use normal convective heat transfer coefficient
        # else use enhanced convective heat transfer coefficient
        
        #hcv_floori = bem_var_con.hcv_roof_enhanced if t_floor_bef >= t_building_bef else bem_var_con.hcv_floor
        if t_floor_bef >= t_building_bef:
            hcv_floori = bem_var_con.hcv_floor_enhanced
        else:
            hcv_floori = bem_var_con.hcv_floor


        hcv_sunwi = bem_var_con.hcv_sunw
        hcv_shdwi = bem_var_con.hcv_shdw

        #* 2 emissivity of interior surfaces
        em_roofi = bem_var_con.em_roof_int
        em_sunwi = bem_var_con.em_sunw_int
        em_shdwi = bem_var_con.em_shdw_int
        em_floori = bem_var_con.em_floor_int

        #* 3 thermal properties of floor
        #* need to input

        #* floor parameters
        # Concrete floor thickness (m)
        dz_floori = bem_var_con.dz_floor
        # Density of concrete floor (kg m-3)
        cp_floori = bem_var_con.cp_floor
        # Specific heat of concrete floor (J kg-1 K-1)
        # Intermediate calculation for concrete floor (W m-2 K-1)
        cv_floori = cp_floori * dz_floori / dtime
        # density of dry air at standard pressure and t_building (kg m-3)
        rair = self.clm_var_con.rair
        pstd = self.clm_var_con.pstd
        sb = self.clm_var_con.sb
        cpair = self.clm_var_con.cpair
        rho_dair = pstd / (rair*t_building_bef)

        # Building height to building width ratio
        #building_hwr = canyon_hwr*(1.0-wtlunit_roof)/wtlunit_roof
        building_hwr = canyon_hwr

        #* 4 building view factors
        vf_rf = np.sqrt(1.0 + canyon_hwr**2.0) - canyon_hwr
        vf_fr = vf_rf
        #! This view factor implicitly converts from per unit wall area to per unit floor area
        vf_wf  = 0.50*(1.0 - vf_rf)
        #! This view factor implicitly converts from per unit floor area to per unit wall area
        vf_fw = vf_wf / canyon_hwr
        #! This view factor implicitly converts from per unit roof area to per unit wall area
        vf_rw  = vf_fw
        #! This view factor implicitly converts from per unit wall area to per unit roof area
        vf_wr  = vf_wf
        vf_ww  = 1.0 - vf_rw - vf_fw

        a = np.zeros((5, 5))
        result = np.zeros(5)

        if (abs(vf_rf+2*vf_wf-1) > 1.0e-6) or\
            (abs(vf_rw+vf_fw+vf_ww-1) > 1.0e-6) or\
            (abs(vf_fr+vf_wr+vf_wr-1) > 1.0e-6):
            raise ValueError("View factors do not sum to 1.0")

        #! ROOF
        a[0,0] =   0.50*hcv_roofi \
                 + 0.50*tk_roof_innerl/(zi_roof_innerl - z_roof_innerl) \
                 + 4.0*em_roofi*sb*t_roof_inner_bef**3.0 \
                 - 4.0*em_roofi*sb*t_roof_inner_bef**3.0*vf_rw*(1.0-em_sunwi)*vf_wr \
                 - 4.0*em_roofi*sb*t_roof_inner_bef**3.0*vf_rw*(1.0-em_shdwi)*vf_wr \
                 - 4.0*em_roofi*sb*t_roof_inner_bef**3.0*vf_rf*(1.0-em_floori)*vf_fr
        #print("hcv_roofi", hcv_roofi)
        #print("tk_roof_innerl", tk_roof_innerl)
        #print("zi_roof_innerl", zi_roof_innerl)
        #print("z_roof_innerl", z_roof_innerl)
        #print("em_roofi", em_roofi)
        #print("sb", sb)
        #print("t_roof_inner_bef", t_roof_inner_bef)
        #print("vf_rw", vf_rw)
        #print("em_sunwi", em_sunwi)
        #print("vf_wr", vf_wr)   
        #print("em_shdwi", em_shdwi)
        #print("vf_wr", vf_wr)
        #print("em_floori", em_floori)
        #print("vf_fr", vf_fr)
        
                 
                 
        a[0,1] = - 4.0*em_roofi*em_sunwi*sb*t_sunw_inner_bef**3.0*vf_wr \
                 - 4.0*em_sunwi*sb*t_sunw_inner_bef**3.0*vf_ww*(1.0-em_shdwi)*vf_wr \
                 - 4.0*em_sunwi*sb*t_sunw_inner_bef**3.0*vf_wf*(1.0-em_floori)*vf_fr
        a[0,2] = - 4.0*em_roofi*em_shdwi*sb*t_shdw_inner_bef**3.0*vf_wr \
                 - 4.0*em_shdwi*sb*t_shdw_inner_bef**3.0*vf_ww*(1.0-em_sunwi)*vf_wr \
                 - 4.0*em_shdwi*sb*t_shdw_inner_bef**3.0*vf_wf*(1.0-em_floori)*vf_fr
        a[0,3] = - 4.0*em_roofi*em_floori*sb*t_floor_bef**3.0*vf_fr \
                 - 4.0*em_floori*sb*t_floor_bef**3.0*vf_fw*(1.0-em_sunwi)*vf_wr \
                 - 4.0*em_floori*sb*t_floor_bef**3.0*vf_fw*(1.0-em_shdwi)*vf_wr
        a[0,4] = - 0.50*hcv_roofi
        result[0] =   0.50*tk_roof_innerl*t_roof_innerl/(zi_roof_innerl - z_roof_innerl) \
                    - 0.50*tk_roof_innerl*(t_roof_inner_bef-t_roof_innerl_bef)/(zi_roof_innerl \
                    - z_roof_innerl) \
                    - 3.0*em_roofi*em_sunwi*sb*t_sunw_inner_bef**4.0*vf_wr \
                    - 3.0*em_roofi*em_shdwi*sb*t_shdw_inner_bef**4.0*vf_wr \
                    - 3.0*em_roofi*em_floori*sb*t_floor_bef**4.0*vf_fr \
                    + 3.0*em_roofi*sb*t_roof_inner_bef**4.0 \
                    - 3.0*em_roofi*sb*t_roof_inner_bef**4.0*vf_rw*(1.0-em_sunwi)*vf_wr \
                    - 3.0*em_roofi*sb*t_roof_inner_bef**4.0*vf_rw*(1.0-em_shdwi)*vf_wr \
                    - 3.0*em_roofi*sb*t_roof_inner_bef**4.0*vf_rf*(1.0-em_floori)*vf_fr \
                    - 3.0*em_sunwi*sb*t_sunw_inner_bef**4.0*vf_ww*(1.0-em_shdwi)*vf_wr \
                    - 3.0*em_sunwi*sb*t_sunw_inner_bef**4.0*vf_wf*(1.0-em_floori)*vf_fr \
                    - 3.0*em_shdwi*sb*t_shdw_inner_bef**4.0*vf_ww*(1.0-em_sunwi)*vf_wr \
                    - 3.0*em_shdwi*sb*t_shdw_inner_bef**4.0*vf_wf*(1.0-em_floori)*vf_fr \
                    - 3.0*em_floori*sb*t_floor_bef**4.0*vf_fw*(1.0-em_sunwi)*vf_wr \
                    - 3.0*em_floori*sb*t_floor_bef**4.0*vf_fw*(1.0-em_shdwi)*vf_wr \
                    - 0.50*hcv_roofi*(t_roof_inner_bef - t_building_bef)
        #! SUNWALL
        a[1,0] = - 4.0*em_sunwi*em_roofi*sb*t_roof_inner_bef**3.0*vf_rw \
                 - 4.0*em_roofi*sb*t_roof_inner_bef**3.0*vf_rw*(1.0-em_shdwi)*vf_ww \
                 - 4.0*em_roofi*sb*t_roof_inner_bef**3.0*vf_rf*(1.0-em_floori)*vf_fw
        #print('em_sunwi', em_sunwi)
        #print('em_roofi', em_roofi)
        #print('sb', sb)
        #print('t_roof_inner_bef', t_roof_inner_bef)
        #print('vf_rw', vf_rw)
        #print('em_shdwi', em_shdwi)
        #print('vf_ww', vf_ww)
        #print('em_floori', em_floori)
        #print('vf_fw', vf_fw)
        
        a[1,1] =   0.50*hcv_sunwi*canyon_hwr \
                 + 0.50*tk_sunw_innerl/(zi_sunw_innerl - z_sunw_innerl)*canyon_hwr \
                 + 4.0*em_sunwi*sb*t_sunw_inner_bef**3.0 \
                 - 4.0*em_sunwi*sb*t_sunw_inner_bef**3.0*vf_wr*(1.0-em_roofi)*vf_rw \
                 - 4.0*em_sunwi*sb*t_sunw_inner_bef**3.0*vf_ww*(1.0-em_shdwi)*vf_ww \
                 - 4.0*em_sunwi*sb*t_sunw_inner_bef**3.0*vf_wf*(1.0-em_floori)*vf_fw
        a[1,2] = - 4.0*em_sunwi*em_shdwi*sb*t_shdw_inner_bef**3.0*vf_ww \
                 - 4.0*em_shdwi*sb*t_shdw_inner_bef**3.0*vf_wf*(1.0-em_floori)*vf_fw \
                 - 4.0*em_shdwi*sb*t_shdw_inner_bef**3.0*vf_wr*(1.0-em_roofi)*vf_rw
        a[1,3] = - 4.0*em_sunwi*em_floori*sb*t_floor_bef**3.0*vf_fw \
                 - 4.0*em_floori*sb*t_floor_bef**3.0*vf_fr*(1.0-em_roofi)*vf_rw \
                 - 4.0*em_floori*sb*t_floor_bef**3.0*vf_fw*(1.0-em_shdwi)*vf_ww
        a[1,4] = - 0.50*hcv_sunwi*canyon_hwr
        result[1] =   0.50*tk_sunw_innerl*t_sunw_innerl/(zi_sunw_innerl - z_sunw_innerl)*canyon_hwr \
                    - 0.50*tk_sunw_innerl*(t_sunw_inner_bef-t_sunw_innerl_bef)/(zi_sunw_innerl \
                    - z_sunw_innerl)*canyon_hwr \
                    - 3.0*em_sunwi*em_roofi*sb*t_roof_inner_bef**4.0*vf_rw \
                    - 3.0*em_sunwi*em_shdwi*sb*t_shdw_inner_bef**4.0*vf_ww \
                    - 3.0*em_sunwi*em_floori*sb*t_floor_bef**4.0*vf_fw \
                    + 3.0*em_sunwi*sb*t_sunw_inner_bef**4.0 \
                    - 3.0*em_sunwi*sb*t_sunw_inner_bef**4.0*vf_wr*(1.0-em_roofi)*vf_rw \
                    - 3.0*em_sunwi*sb*t_sunw_inner_bef**4.0*vf_ww*(1.0-em_shdwi)*vf_ww \
                    - 3.0*em_sunwi*sb*t_sunw_inner_bef**4.0*vf_wf*(1.0-em_floori)*vf_fw \
                    - 3.0*em_shdwi*sb*t_shdw_inner_bef**4.0*vf_wf*(1.0-em_floori)*vf_fw \
                    - 3.0*em_shdwi*sb*t_shdw_inner_bef**4.0*vf_wr*(1.0-em_roofi)*vf_rw \
                    - 3.0*em_roofi*sb*t_roof_inner_bef**4.0*vf_rw*(1.0-em_shdwi)*vf_ww \
                    - 3.0*em_roofi*sb*t_roof_inner_bef**4.0*vf_rf*(1.0-em_floori)*vf_fw \
                    - 3.0*em_floori*sb*t_floor_bef**4.0*vf_fr*(1.0-em_roofi)*vf_rw \
                    - 3.0*em_floori*sb*t_floor_bef**4.0*vf_fw*(1.0-em_shdwi)*vf_ww \
                    - 0.50*hcv_sunwi*(t_sunw_inner_bef - t_building_bef)*canyon_hwr
        #! SHADEWALL
        a[2,0] = - 4.0*em_shdwi*em_roofi*sb*t_roof_inner_bef**3.0*vf_rw \
                 - 4.0*em_roofi*sb*t_roof_inner_bef**3.0*vf_rw*(1.0-em_sunwi)*vf_ww \
                 - 4.0*em_roofi*sb*t_roof_inner_bef**3.0*vf_rf*(1.0-em_floori)*vf_fw
        a[2,1] = - 4.0*em_shdwi*em_sunwi*sb*t_sunw_inner_bef**3.0*vf_ww \
                 - 4.0*em_sunwi*sb*t_sunw_inner_bef**3.0*vf_wf*(1.0-em_floori)*vf_fw \
                 - 4.0*em_sunwi*sb*t_sunw_inner_bef**3.0*vf_wr*(1.0-em_roofi)*vf_rw
        a[2,2] =   0.50*hcv_shdwi*canyon_hwr \
                 + 0.50*tk_shdw_innerl/(zi_shdw_innerl - z_shdw_innerl)*canyon_hwr \
                 + 4.0*em_shdwi*sb*t_shdw_inner_bef**3.0 \
                 - 4.0*em_shdwi*sb*t_shdw_inner_bef**3.0*vf_wr*(1.0-em_roofi)*vf_rw \
                 - 4.0*em_shdwi*sb*t_shdw_inner_bef**3.0*vf_ww*(1.0-em_sunwi)*vf_ww \
                 - 4.0*em_shdwi*sb*t_shdw_inner_bef**3.0*vf_wf*(1.0-em_floori)*vf_fw
        a[2,3] = - 4.0*em_shdwi*em_floori*sb*t_floor_bef**3.0*vf_fw \
                 - 4.0*em_floori*sb*t_floor_bef**3.0*vf_fr*(1.0-em_roofi)*vf_rw \
                 - 4.0*em_floori*sb*t_floor_bef**3.0*vf_fw*(1.0-em_sunwi)*vf_ww
        a[2,4] = - 0.50*hcv_shdwi*canyon_hwr
        result[2] =   0.50*tk_shdw_innerl*t_shdw_innerl/(zi_shdw_innerl - z_shdw_innerl)*canyon_hwr \
                    - 0.50*tk_shdw_innerl*(t_shdw_inner_bef-t_shdw_innerl_bef)/(zi_shdw_innerl \
                    - z_shdw_innerl)*canyon_hwr \
                    - 3.0*em_shdwi*em_roofi*sb*t_roof_inner_bef**4.0*vf_rw \
                    - 3.0*em_shdwi*em_sunwi*sb*t_sunw_inner_bef**4.0*vf_ww \
                    - 3.0*em_shdwi*em_floori*sb*t_floor_bef**4.0*vf_fw \
                    + 3.0*em_shdwi*sb*t_shdw_inner_bef**4.0 \
                    - 3.0*em_shdwi*sb*t_shdw_inner_bef**4.0*vf_wr*(1.0-em_roofi)*vf_rw \
                    - 3.0*em_shdwi*sb*t_shdw_inner_bef**4.0*vf_ww*(1.0-em_sunwi)*vf_ww \
                    - 3.0*em_shdwi*sb*t_shdw_inner_bef**4.0*vf_wf*(1.0-em_floori)*vf_fw \
                    - 3.0*em_sunwi*sb*t_sunw_inner_bef**4.0*vf_wf*(1.0-em_floori)*vf_fw \
                    - 3.0*em_sunwi*sb*t_sunw_inner_bef**4.0*vf_wr*(1.0-em_roofi)*vf_rw \
                    - 3.0*em_roofi*sb*t_roof_inner_bef**4.0*vf_rw*(1.0-em_sunwi)*vf_ww \
                    - 3.0*em_roofi*sb*t_roof_inner_bef**4.0*vf_rf*(1.0-em_floori)*vf_fw \
                    - 3.0*em_floori*sb*t_floor_bef**4.0*vf_fr*(1.0-em_roofi)*vf_rw \
                    - 3.0*em_floori*sb*t_floor_bef**4.0*vf_fw*(1.0-em_sunwi)*vf_ww \
                    - 0.50*hcv_shdwi*(t_shdw_inner_bef - t_building_bef)*canyon_hwr
        #! FLOOR
        a[3,0] = - 4.0*em_floori*em_roofi*sb*t_roof_inner_bef**3.0*vf_rf \
                 - 4.0*em_roofi*sb*t_roof_inner_bef**3.0*vf_rw*(1.0-em_sunwi)*vf_wf \
                 - 4.0*em_roofi*sb*t_roof_inner_bef**3.0*vf_rw*(1.0-em_shdwi)*vf_wf
        a[3,1] = - 4.0*em_floori*em_sunwi*sb*t_sunw_inner_bef**3.0*vf_wf \
                 - 4.0*em_sunwi*sb*t_sunw_inner_bef**3.0*vf_ww*(1.0-em_shdwi)*vf_wf \
                 - 4.0*em_sunwi*sb*t_sunw_inner_bef**3.0*vf_wr*(1.0-em_roofi)*vf_rf
        a[3,2] = - 4.0*em_floori*em_shdwi*sb*t_shdw_inner_bef**3.0*vf_wf \
                 - 4.0*em_shdwi*sb*t_shdw_inner_bef**3.0*vf_wr*(1.0-em_roofi)*vf_rf \
                 - 4.0*em_shdwi*sb*t_shdw_inner_bef**3.0*vf_ww*(1.0-em_sunwi)*vf_wf
        a[3,3] =   (cv_floori + 0.50*hcv_floori) \
                 + 4.0*em_floori*sb*t_floor_bef**3.0 \
                 - 4.0*em_floori*sb*t_floor_bef**3.0*vf_fr*(1.0-em_roofi)*vf_rf \
                 - 4.0*em_floori*sb*t_floor_bef**3.0*vf_fw*(1.0-em_sunwi)*vf_wf \
                 - 4.0*em_floori*sb*t_floor_bef**3.0*vf_fw*(1.0-em_shdwi)*vf_wf
        a[3,4] = - 0.50*hcv_floori
        result[3] =   cv_floori*t_floor_bef \
                    - 3.0*em_floori*em_roofi*sb*t_roof_inner_bef**4.0*vf_rf \
                    - 3.0*em_floori*em_sunwi*sb*t_sunw_inner_bef**4.0*vf_wf \
                    - 3.0*em_floori*em_shdwi*sb*t_shdw_inner_bef**4.0*vf_wf \
                    + 3.0*em_floori*sb*t_floor_bef**4.0 \
                    - 3.0*em_floori*sb*t_floor_bef**4.0*vf_fr*(1.0-em_roofi)*vf_rf \
                    - 3.0*em_floori*sb*t_floor_bef**4.0*vf_fw*(1.0-em_sunwi)*vf_wf \
                    - 3.0*em_floori*sb*t_floor_bef**4.0*vf_fw*(1.0-em_shdwi)*vf_wf \
                    - 3.0*em_sunwi*sb*t_sunw_inner_bef**4.0*vf_ww*(1.0-em_shdwi)*vf_wf \
                    - 3.0*em_sunwi*sb*t_sunw_inner_bef**4.0*vf_wr*(1.0-em_roofi)*vf_rf \
                    - 3.0*em_shdwi*sb*t_shdw_inner_bef**4.0*vf_wr*(1.0-em_roofi)*vf_rf \
                    - 3.0*em_shdwi*sb*t_shdw_inner_bef**4.0*vf_ww*(1.0-em_sunwi)*vf_wf \
                    - 3.0*em_roofi*sb*t_roof_inner_bef**4.0*vf_rw*(1.0-em_sunwi)*vf_wf \
                    - 3.0*em_roofi*sb*t_roof_inner_bef**4.0*vf_rw*(1.0-em_shdwi)*vf_wf \
                    - 0.50*hcv_floori*(t_floor_bef - t_building_bef)
        #! Building air temperature
        a[4,0] = - 0.50*hcv_roofi
        a[4,1] = - 0.50*hcv_sunwi*canyon_hwr
        a[4,2] = - 0.50*hcv_shdwi*canyon_hwr
        a[4,3] = - 0.50*hcv_floori
        a[4,4] =  ((ht_roof*rho_dair*cpair)/dtime) + \
                  ((ht_roof*vent_ach)/3600.0)*rho_dair*cpair + \
                  0.50*hcv_roofi + \
                  0.50*hcv_sunwi*canyon_hwr + \
                  0.50*hcv_shdwi*canyon_hwr + \
                  0.50*hcv_floori
        result[4] = (ht_roof*rho_dair*cpair/dtime)*t_building_bef \
                     + ((ht_roof*vent_ach)/3600.0)*rho_dair*cpair*taf \
                     + 0.50*hcv_roofi*(t_roof_inner_bef - t_building_bef) \
                     + 0.50*hcv_sunwi*(t_sunw_inner_bef - t_building_bef)*canyon_hwr \
                     + 0.50*hcv_shdwi*(t_shdw_inner_bef - t_building_bef)*canyon_hwr \
                     + 0.50*hcv_floori*(t_floor_bef - t_building_bef)
        #print("result 5")
        #print('ht_roof', ht_roof)
        #print('rho_dair', rho_dair)
        #print('cpair', cpair)
        #print('dtime', dtime)
        #print('vent_ach', vent_ach)
        #print('t_building_bef', t_building_bef)
        #print('taf', taf)
        #print('t_roof_inner_bef', t_roof_inner_bef)
        #print('t_sunw_inner_bef', t_sunw_inner_bef)
        #print('t_shdw_inner_bef', t_shdw_inner_bef)
        #print('t_floor_bef', t_floor_bef)
        #print('t_building_bef', t_building_bef)
        #print('hcv_roofi', hcv_roofi)
        #print('hcv_sunwi', hcv_sunwi)
        #print('hcv_shdwi', hcv_shdwi)
        #print('hcv_floori', hcv_floori)
        #print('hcv_floori', hcv_floori)
        #print('hcv_floori', hcv_floori)
        #print("result 5")
        #print(a)
        #print(result)
        # Solve equations
        solution = np.linalg.solve(a, result)
        #lu, piv, solution, info = dgesv(a, result)
        # Update inner temperatures and room temperature
        t_roof_inner = solution[0]
        t_sunw_inner = solution[1]
        t_shdw_inner = solution[2]
        t_floor = solution[3]
        t_building = solution[4]
        
        #print(t_roof_inner_bef, t_sunw_inner_bef, t_shdw_inner_bef, t_floor_bef, t_building_bef)
        #print(t_roof_inner, t_sunw_inner, t_shdw_inner, t_floor, t_building)
        #print(rho_dair)
        
        #qrd_roof = - em_roofi*em_sunwi*sb*t_sunw_inner_bef**4.0*vf_wr \
        #               - 4.0*em_roofi*em_sunwi*sb*t_sunw_inner_bef**3.0*vf_wr*(t_sunw_inner \
        #               - t_sunw_inner_bef) \
        #               - em_roofi*em_shdwi*sb*t_shdw_inner_bef**4.0*vf_wr \
        #               - 4.0*em_roofi*em_shdwi*sb*t_shdw_inner_bef**3.0*vf_wr*(t_shdw_inner \
        #               - t_shdw_inner_bef) \
        #               - em_roofi*em_floori*sb*t_floor_bef**4.0*vf_fr \
        #               - 4.0*em_roofi*em_floori*sb*t_floor_bef**3.0*vf_fr*(t_floor - t_floor_bef) \
        #               - (em_roofi*sb*t_roof_inner_bef**4.0)*vf_rw*(1.0-em_sunwi)*vf_wr \
        #               - 4.0*em_roofi*sb*t_roof_inner_bef**3.0*vf_rw*(1.0-em_sunwi)*vf_wr*(t_roof_inner \
        #               - t_roof_inner_bef) \
        #               - (em_roofi*sb*t_roof_inner_bef**4.0)*vf_rw*(1.0-em_shdwi)*vf_wr \
        #               - 4.0*em_roofi*sb*t_roof_inner_bef**3.0*vf_rw*(1.0-em_shdwi)*vf_wr*(t_roof_inner \
        #               - t_roof_inner_bef) \
        #               - (em_roofi*sb*t_roof_inner_bef**4.0)*vf_rf*(1.0-em_floori)*vf_fr \
        #               - 4.0*em_roofi*sb*t_roof_inner_bef**3.0*vf_rf*(1.0-em_floori)*vf_fr*(t_roof_inner \
        #               - t_roof_inner_bef) \
        #               - (em_sunwi*sb*t_sunw_inner_bef**4.0)*vf_ww*(1.0-em_shdwi)*vf_wr \
        #               - 4.0*em_sunwi*sb*t_sunw_inner_bef**3.0*vf_ww*(1.0-em_shdwi)*vf_wr*(t_sunw_inner \
        #               - t_sunw_inner_bef) \
        #               - (em_sunwi*sb*t_sunw_inner_bef**4.0)*vf_wf*(1.0-em_floori)*vf_fr \
        #               - 4.0*em_sunwi*sb*t_sunw_inner_bef**3.0*vf_wf*(1.0-em_floori)*vf_fr*(t_sunw_inner \
        #               - t_sunw_inner_bef) \
        #               - (em_shdwi*sb*t_shdw_inner_bef**4.0)*vf_ww*(1.0-em_sunwi)*vf_wr \
        #               - 4.0*em_shdwi*sb*t_shdw_inner_bef**3.0*vf_ww*(1.0-em_sunwi)*vf_wr*(t_shdw_inner \
        #               - t_shdw_inner_bef) \
        #               - (em_shdwi*sb*t_shdw_inner_bef**4.0)*vf_wf*(1.0-em_floori)*vf_fr \
        #               - 4.0*em_shdwi*sb*t_shdw_inner_bef**3.0*vf_wf*(1.0-em_floori)*vf_fr*(t_shdw_inner \
        #               - t_shdw_inner_bef) \
        #               - (em_floori*sb*t_floor_bef**4.0)*vf_fw*(1.0-em_sunwi)*vf_wr \
        #               - 4.0*em_floori*sb*t_floor_bef**3.0*vf_fw*(1.0-em_sunwi)*vf_wr*(t_floor \
        #               - t_floor_bef) \
        #               - (em_floori*sb*t_floor_bef**4.0)*vf_fw*(1.0-em_shdwi)*vf_wr \
        #               - 4.0*em_floori*sb*t_floor_bef**3.0*vf_fw*(1.0-em_shdwi)*vf_wr*(t_floor \
        #               - t_floor_bef) \
        #               + em_roofi*sb*t_roof_inner_bef**4.0 \
        #               + 4.0*em_roofi*sb*t_roof_inner_bef**3.0*(t_roof_inner - t_roof_inner_bef) 
#
        #qrd_sunw = - em_sunwi*em_roofi*sb*t_roof_inner_bef**4.0*vf_rw \
        #               - 4.0*em_sunwi*em_roofi*sb*t_roof_inner_bef**3.0*vf_rw*(t_roof_inner \
        #               - t_roof_inner_bef) \
        #               - em_sunwi*em_shdwi*sb*t_shdw_inner_bef**4.0*vf_ww  \
        #               - 4.0*em_sunwi*em_shdwi*sb*t_shdw_inner_bef**3.0*vf_ww*(t_shdw_inner \
        #               - t_shdw_inner_bef) \
        #               - em_sunwi*em_floori*sb*t_floor_bef**4.0*vf_fw \
        #               - 4.0*em_sunwi*em_floori*sb*t_floor_bef**3.0*vf_fw*(t_floor - t_floor_bef) \
        #               - (em_sunwi*sb*t_sunw_inner_bef**4.0)*vf_wr*(1.0-em_roofi)*vf_rw \
        #               - 4.0*em_sunwi*sb*t_sunw_inner_bef**3.*vf_wr*(1.0-em_roofi)*vf_rw*(t_sunw_inner \
        #               - t_sunw_inner_bef) \
        #               - (em_sunwi*sb*t_sunw_inner_bef**4.0)*vf_ww*(1.0-em_shdwi)*vf_ww \
        #               - 4.0*em_sunwi*sb*t_sunw_inner_bef**3.*vf_ww*(1.0-em_shdwi)*vf_ww*(t_sunw_inner \
        #               - t_sunw_inner_bef) \
        #               - (em_sunwi*sb*t_sunw_inner_bef**4.0)*vf_wf*(1.0-em_floori)*vf_fw \
        #               - 4.0*em_sunwi*sb*t_sunw_inner_bef**3.*vf_wf*(1.0-em_floori)*vf_fw*(t_sunw_inner \
        #               - t_sunw_inner_bef) \
        #               - (em_shdwi*sb*t_shdw_inner_bef**4.0)*vf_wf*(1.0-em_floori)*vf_fw \
        #               - 4.0*em_shdwi*sb*t_shdw_inner_bef**3.*vf_wf*(1.0-em_floori)*vf_fw*(t_shdw_inner \
        #               - t_shdw_inner_bef) \
        #               - (em_shdwi*sb*t_shdw_inner_bef**4.0)*vf_wr*(1.0-em_roofi)*vf_rw \
        #               - 4.0*em_shdwi*sb*t_shdw_inner_bef**3.*vf_wr*(1.0-em_roofi)*vf_rw*(t_shdw_inner \
        #               - t_shdw_inner_bef) \
        #               - (em_roofi*sb*t_roof_inner_bef**4.0)*vf_rw*(1.0-em_shdwi)*vf_ww \
        #               - 4.0*em_roofi*sb*t_roof_inner_bef**3.*vf_rw*(1.0-em_shdwi)*vf_ww*(t_roof_inner \
        #               - t_roof_inner_bef) \
        #               - (em_roofi*sb*t_roof_inner_bef**4.0)*vf_rf*(1.0-em_floori)*vf_fw \
        #               - 4.0*em_roofi*sb*t_roof_inner_bef**3.*vf_rf*(1.0-em_floori)*vf_fw*(t_roof_inner \
        #               - t_roof_inner_bef) \
        #               - (em_floori*sb*t_floor_bef**4.0)*vf_fr*(1.0-em_roofi)*vf_rw \
        #               - 4.0*em_floori*sb*t_floor_bef**3.*vf_fr*(1.0-em_roofi)*vf_rw*(t_floor \
        #               - t_floor_bef) \
        #               - (em_floori*sb*t_floor_bef**4.0)*vf_fw*(1.0-em_shdwi)*vf_ww \
        #               - 4.0*em_floori*sb*t_floor_bef**3.*vf_fw*(1.0-em_shdwi)*vf_ww*(t_floor \
        #               - t_floor_bef) \
        #               + em_sunwi*sb*t_sunw_inner_bef**4.0 \
        #               + 4.0*em_sunwi*sb*t_sunw_inner_bef**3.0*(t_sunw_inner - t_sunw_inner_bef)
#
        #
        #qrd_shdw = - em_shdwi*em_roofi*sb*t_roof_inner_bef**4.0*vf_rw \
        #               - 4.0*em_shdwi*em_roofi*sb*t_roof_inner_bef**3.0*vf_rw*(t_roof_inner \
        #               - t_roof_inner_bef) \
        #               - em_shdwi*em_sunwi*sb*t_sunw_inner_bef**4.0*vf_ww \
        #               - 4.0*em_shdwi*em_sunwi*sb*t_sunw_inner_bef**3.0*vf_ww*(t_sunw_inner \
        #               - t_sunw_inner_bef) \
        #               - em_shdwi*em_floori*sb*t_floor_bef**4.0*vf_fw \
        #               - 4.0*em_shdwi*em_floori*sb*t_floor_bef**3.0*vf_fw*(t_floor - t_floor_bef) \
        #               - (em_shdwi*sb*t_shdw_inner_bef**4.0)*vf_wr*(1.0-em_roofi)*vf_rw \
        #               - 4.0*em_shdwi*sb*t_shdw_inner_bef**3.*vf_wr*(1.0-em_roofi)*vf_rw*(t_shdw_inner \
        #               - t_shdw_inner_bef) \
        #               - (em_shdwi*sb*t_shdw_inner_bef**4.0)*vf_ww*(1.0-em_sunwi)*vf_ww \
        #               - 4.0*em_shdwi*sb*t_shdw_inner_bef**3.*vf_ww*(1.0-em_sunwi)*vf_ww*(t_shdw_inner \
        #               - t_shdw_inner_bef) \
        #               - (em_shdwi*sb*t_shdw_inner_bef**4.0)*vf_wf*(1.0-em_floori)*vf_fw \
        #               - 4.0*em_shdwi*sb*t_shdw_inner_bef**3.*vf_wf*(1.0-em_floori)*vf_fw*(t_shdw_inner \
        #               - t_shdw_inner_bef) \
        #               - (em_sunwi*sb*t_sunw_inner_bef**4.0)*vf_wf*(1.0-em_floori)*vf_fw \
        #               - 4.0*em_sunwi*sb*t_sunw_inner_bef**3.*vf_wf*(1.0-em_floori)*vf_fw*(t_sunw_inner \
        #               - t_sunw_inner_bef) \
        #               - (em_sunwi*sb*t_sunw_inner_bef**4.0)*vf_wr*(1.0-em_roofi)*vf_rw \
        #               - 4.0*em_sunwi*sb*t_sunw_inner_bef**3.*vf_wr*(1.0-em_roofi)*vf_rw*(t_sunw_inner \
        #               - t_sunw_inner_bef) \
        #               - (em_roofi*sb*t_roof_inner_bef**4.0)*vf_rw*(1.0-em_sunwi)*vf_ww \
        #               - 4.0*em_roofi*sb*t_roof_inner_bef**3.*vf_rw*(1.0-em_sunwi)*vf_ww*(t_roof_inner \
        #               - t_roof_inner_bef) \
        #               - (em_roofi*sb*t_roof_inner_bef**4.0)*vf_rf*(1.0-em_floori)*vf_fw \
        #               - 4.0*em_roofi*sb*t_roof_inner_bef**3.*vf_rf*(1.0-em_floori)*vf_fw*(t_roof_inner \
        #               - t_roof_inner_bef) \
        #               - (em_floori*sb*t_floor_bef**4.0)*vf_fr*(1.0-em_roofi)*vf_rw \
        #               - 4.0*em_floori*sb*t_floor_bef**3.*vf_fr*(1.0-em_roofi)*vf_rw*(t_floor \
        #               - t_floor_bef) \
        #               - (em_floori*sb*t_floor_bef**4.0)*vf_fw*(1.0-em_sunwi)*vf_ww \
        #               - 4.0*em_floori*sb*t_floor_bef**3.*vf_fw*(1.0-em_sunwi)*vf_ww*(t_floor \
        #               - t_floor_bef) \
        #               + em_shdwi*sb*t_shdw_inner_bef**4.0 \
        #               + 4.0*em_shdwi*sb*t_shdw_inner_bef**3.0*(t_shdw_inner - t_shdw_inner_bef)
#
        #qrd_floor = - em_floori*em_roofi*sb*t_roof_inner_bef**4.0*vf_rf \
        #                - 4.0*em_floori*em_roofi*sb*t_roof_inner_bef**3.0*vf_rf*(t_roof_inner \
        #                - t_roof_inner_bef) \
        #                - em_floori*em_sunwi*sb*t_sunw_inner_bef**4.0*vf_wf \
        #                - 4.0*em_floori*em_sunwi*sb*t_sunw_inner_bef**3.0*vf_wf*(t_sunw_inner \
        #                - t_sunw_inner_bef) \
        #                - em_floori*em_shdwi*sb*t_shdw_inner_bef**4.0*vf_wf \
        #                - 4.0*em_floori*em_shdwi*sb*t_shdw_inner_bef**3.0*vf_wf*(t_shdw_inner \
        #                - t_shdw_inner_bef) \
        #                - (em_floori*sb*t_floor_bef**4.0)*vf_fr*(1.0-em_roofi)*vf_rf \
        #                - 4.0*em_floori*sb*t_floor_bef**3.*vf_fr*(1.0-em_roofi)*vf_rf*(t_floor \
        #                - t_floor_bef) \
        #                - (em_floori*sb*t_floor_bef**4.0)*vf_fw*(1.0-em_sunwi)*vf_wf \
        #                - 4.0*em_floori*sb*t_floor_bef**3.*vf_fw*(1.0-em_sunwi)*vf_wf*(t_floor \
        #                - t_floor_bef) \
        #                - (em_floori*sb*t_floor_bef**4.0)*vf_fw*(1.0-em_shdwi)*vf_wf \
        #                - 4.0*em_floori*sb*t_floor_bef**3.*vf_fw*(1.0-em_shdwi)*vf_wf*(t_floor \
        #                - t_floor_bef) \
        #                - (em_sunwi*sb*t_sunw_inner_bef**4.0)*vf_ww*(1.0-em_shdwi)*vf_wf \
        #                - 4.0*em_sunwi*sb*t_sunw_inner_bef**3.*vf_ww*(1.0-em_shdwi)*vf_wf*(t_sunw_inner \
        #                - t_sunw_inner_bef) \
        #                - (em_sunwi*sb*t_sunw_inner_bef**4.0)*vf_wr*(1.0-em_roofi)*vf_rf \
        #                - 4.0*em_sunwi*sb*t_sunw_inner_bef**3.*vf_wr*(1.0-em_roofi)*vf_rf*(t_sunw_inner \
        #                - t_sunw_inner_bef) \
        #                - (em_shdwi*sb*t_shdw_inner_bef**4.0)*vf_wr*(1.0-em_roofi)*vf_rf \
        #                - 4.0*em_shdwi*sb*t_shdw_inner_bef**3.*vf_wr*(1.0-em_roofi)*vf_rf*(t_shdw_inner \
        #                - t_shdw_inner_bef) \
        #                - (em_shdwi*sb*t_shdw_inner_bef**4.0)*vf_ww*(1.0-em_sunwi)*vf_wf \
        #                - 4.0*em_shdwi*sb*t_shdw_inner_bef**3.*vf_ww*(1.0-em_sunwi)*vf_wf*(t_shdw_inner \
        #                - t_shdw_inner_bef) \
        #                - (em_roofi*sb*t_roof_inner_bef**4.0)*vf_rw*(1.0-em_sunwi)*vf_wf \
        #                - 4.0*em_roofi*sb*t_roof_inner_bef**3.*vf_rw*(1.0-em_sunwi)*vf_wf*(t_roof_inner \
        #                - t_roof_inner_bef) \
        #                - (em_roofi*sb*t_roof_inner_bef**4.0)*vf_rw*(1.0-em_shdwi)*vf_wf \
        #                - 4.0*em_roofi*sb*t_roof_inner_bef**3.*vf_rw*(1.0-em_shdwi)*vf_wf*(t_roof_inner \
        #                - t_roof_inner_bef) \
        #                + em_floori*sb*t_floor_bef**4.0 \
        #                + 4.0*em_floori*sb*t_floor_bef**3.*(t_floor - t_floor_bef)
#
        #qrd_building = qrd_roof + qrd_sunw + qrd_shdw + qrd_floor
        #
        #qcv_roof = 0.50*hcv_roofi*(t_roof_inner - t_building) + 0.50*hcv_roofi*(t_roof_inner_bef \
        #               - t_building_bef)
        #qcd_roof = 0.50*tk_roof_innerl*(t_roof_inner - t_roof_innerl)/(zi_roof_innerl - z_roof_innerl)  \
        #               + 0.50*tk_roof_innerl*(t_roof_inner_bef - t_roof_innerl_bef)/(zi_roof_innerl \
        #               - z_roof_innerl)
        #enrgy_bal_roof = qrd_roof + qcv_roof + qcd_roof
        #
        #qcv_sunw = 0.50*hcv_sunwi*(t_sunw_inner - t_building) + 0.50*hcv_sunwi*(t_sunw_inner_bef \
        #               - t_building_bef)
        #qcd_sunw = 0.50*tk_sunw_innerl*(t_sunw_inner - t_sunw_innerl)/(zi_sunw_innerl - z_sunw_innerl)  \
        #               + 0.50*tk_sunw_innerl*(t_sunw_inner_bef - t_sunw_innerl_bef)/(zi_sunw_innerl \
        #               - z_sunw_innerl)
        #enrgy_bal_sunw = qrd_sunw + qcv_sunw*building_hwr + qcd_sunw*building_hwr
        #
        #qcv_shdw = 0.50*hcv_shdwi*(t_shdw_inner - t_building) + 0.50*hcv_shdwi*(t_shdw_inner_bef \
        #               - t_building_bef)
        #qcd_shdw = 0.50*tk_shdw_innerl*(t_shdw_inner - t_shdw_innerl)/(zi_shdw_innerl - z_shdw_innerl)  \
        #               + 0.50*tk_shdw_innerl*(t_shdw_inner_bef - t_shdw_innerl_bef)/(zi_shdw_innerl \
        #               - z_shdw_innerl)
        #enrgy_bal_shdw = qrd_shdw + qcv_shdw*building_hwr + qcd_shdw*building_hwr
        #
        #qcv_floor = 0.50*hcv_floori*(t_floor - t_building) + 0.50*hcv_floori*(t_floor_bef \
        #                - t_building_bef)
        #qcd_floor = cv_floori*(t_floor - t_floor_bef)
        #enrgy_bal_floor = qrd_floor + qcv_floor + qcd_floor
        #
        #enrgy_bal_buildair = (ht_roof*rho_dair*cpair/dtime)*(t_building - t_building_bef) \
        #                         - ht_roof*(vent_ach/3600.0)*rho_dair*cpair*(taf - t_building) \
        #                         - 0.50*hcv_roofi*(t_roof_inner - t_building) \
        #                         - 0.50*hcv_roofi*(t_roof_inner_bef - t_building_bef) \
        #                         - 0.50*hcv_sunwi*(t_sunw_inner - t_building)*building_hwr \
        #                         - 0.50*hcv_sunwi*(t_sunw_inner_bef - t_building_bef)*building_hwr \
        #                         - 0.50*hcv_shdwi*(t_shdw_inner - t_building)*building_hwr \
        #                         - 0.50*hcv_shdwi*(t_shdw_inner_bef - t_building_bef)*building_hwr \
        #                         - 0.50*hcv_floori*(t_floor - t_building) \
        #                         - 0.50*hcv_floori*(t_floor_bef - t_building_bef)
        #                         
        #if (abs(enrgy_bal_roof) > 1.0e-6) or (abs(enrgy_bal_sunw) > 1.0e-6) or (abs(enrgy_bal_shdw) > 1.0e-6) \
        #    or (abs(enrgy_bal_floor) > 1.0e-6) or (abs(enrgy_bal_buildair) > 1.0e-6) or (abs(qrd_building) > 1.0e-6):
        #    raise ValueError("Energy balance not satisfied.")
        

        # Assuming necessary variables are already defined (urban_hac, t_building, pstd, rair, etc.)
        # Some operations like array manipulation or slicing are assumed to be similar in Python.
        urban_hac_on = "on"
        urban_wasteheat_on = "wasteheat"
        eflx_urban_ac = 0.0
        eflx_urban_heat = 0.0
        if (urban_hac.strip() == urban_hac_on) or (urban_hac.strip() == urban_wasteheat_on):
            t_building_bef_hac = t_building
            # rho_dair = pstd / (rair * t_building)  # Uncomment this if necessary

            if t_building_bef_hac > t_building_max:
                if urban_explicit_ac:
                    # Explicit AC adoption rate parameterization scheme
                    eflx_urban_ac_sat = wtlunit_roof * abs(
                        (ht_roof * rho_dair * cpair / dtime) * t_building_max
                        - (ht_roof * rho_dair * cpair / dtime) * t_building_bef_hac
                    )
                    t_building = t_building_max + (1.0 - p_ac) * eflx_urban_ac_sat * dtime / (
                        ht_roof * rho_dair * cpair * wtlunit_roof
                    )
                    eflx_urban_ac = p_ac * eflx_urban_ac_sat
                else:
                    t_building = t_building_max
                    eflx_urban_ac = wtlunit_roof * abs(
                        (ht_roof * rho_dair * cpair / dtime) * t_building
                        - (ht_roof * rho_dair * cpair / dtime) * t_building_bef_hac
                    )
            elif t_building_bef_hac < t_building_min:
                t_building = t_building_min
                eflx_urban_heat = wtlunit_roof * abs(
                    (ht_roof * rho_dair * cpair / dtime) * t_building
                    - (ht_roof * rho_dair * cpair / dtime) * t_building_bef_hac
                )
            else:
                eflx_urban_ac = 0.0
                eflx_urban_heat = 0.0
        
        elif urban_hac.strip() == "heat":
            eflx_urban_heat = wtlunit_roof * abs(
                    (ht_roof * rho_dair * cpair / dtime) * t_building
                    - (ht_roof * rho_dair * cpair / dtime) * t_building_bef_hac
                )
        
        elif urban_hac.strip() == "ac":
            eflx_urban_ac = wtlunit_roof * abs(
                    (ht_roof * rho_dair * cpair / dtime) * t_building
                    - (ht_roof * rho_dair * cpair / dtime) * t_building_bef_hac
                )
        
        else:
            eflx_urban_ac = 0.0
            eflx_urban_heat = 0.0

        eflx_building = wtlunit_roof * (ht_roof * rho_dair * cpair / dtime) * (t_building - t_building_bef)

        eflx_ventilation = wtlunit_roof * ( - ht_roof*(vent_ach/3600.0) \
                               * rho_dair * cpair * (taf - t_building) )
        
        
        info = {
            "t_roof_inner [K]": t_roof_inner,
            "t_sunw_inner [K]": t_sunw_inner,
            "t_shdw_inner [K]": t_shdw_inner,
            "t_floor [K]": t_floor,
            "t_building [K]": t_building,
            "eflx_building [W/m**2]": eflx_building,
            "eflx_urban_ac [W/m**2]": eflx_urban_ac,
            "eflx_urban_heat [W/m**2]": eflx_urban_heat,
            "eflx_ventilation [W/m**2]": eflx_ventilation
        }
        
        return t_roof_inner, t_sunw_inner, t_shdw_inner, t_floor, t_building, info

