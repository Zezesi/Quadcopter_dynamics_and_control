'''
A quadcopter with 4 propellers, the center of the moving body frame i is fixed with the c.o.g of the quadcopter,
the center of the fixed inertia frame is fixed at the initial position of the quadcopter.
By using Euler angles techniques, it is possible to mathematically model the dynamics of the quadcopter,
assuming that the quadcopter always rotates in a same order in the mathematical model of dynamics,
even though in reality, the quadcopter doesn't move exactly in this way or this order.
Most models use the rotation order zyx(of the moving body frame),
I will use the rotation order xyz(of the moving body frame) just because I want,
but make sure that the rotation order you choose is uniform throughout the project.
Two frames: moving (body) frame i and fixed (inertia) frame 0.
'''
import numpy as np
import matplotlib.pyplot as plt
Ts=0.1 # sampling time(s)
k=3.7102*1e-5 # thrust coefficient(N*s^2/rad^2)
b=7.6933*1e-7 # torque coefficient(Nm*s^2/rad^2)
l=0.243 # arm length(m)
alpha=np.pi/4 # angle between the arm and the y axis of the body frame
ag=np.array([[0],[0],[-9.81]]) # gravitational acceleration in the fixed frame(m/s^2)
m=1.587 # total mass(kg)
Ax=0.25 # translational aerodynamic coefficient in the x direction of the body frame(kg/s)
Ay=0.25 # translational aerodynamic coefficient in the y direction of the body frame(kg/s)
Az=0.25 # translational aerodynamic coefficient in the z direction of the body frame(kg/s)
Ar=0.25 # rotational aerodynamic coefficient(kg*m^2/s)
Ixx=0.0213 # mass moment of inertia around x axis in the body frame(kg*m^2)
Iyy=0.02217 # mass moment of inertia around y axis in the body frame(kg*m^2)
Izz=0.0282 # mass moment of inertia around y axis in the body frame(kg*m^2)
Ir=3.357*1e-5 # rotor mass moment of inertia(kg*m^2)
h=0.07 # quadcopter height
def rotation_matrix0i(r,p,r_pre,p_pre,yaw_pre):
    R0i=np.array([[np.cos(p_pre)*np.cos(yaw_pre)                                               , 0         , np.sin(p)],
                  [np.sin(p_pre)*np.sin(r_pre)*np.cos(yaw_pre) + np.sin(yaw_pre)*np.cos(r_pre) , np.cos(r) , -np.sin(r)*np.cos(p)],
                  [-np.sin(p_pre)*np.cos(r_pre)*np.cos(yaw_pre) + np.sin(r_pre)*np.sin(yaw_pre), np.sin(r) , np.cos(r)*np.cos(p)]])
    return R0i


def transformation_matrix0i(x,y,z,r,p,yaw):
    t0i=np.array([[np.cos(p)*np.cos(yaw)                                 , -np.sin(yaw)*np.cos(p)                                  , np.sin(p)           , x],
                  [np.sin(p)*np.sin(r)*np.cos(yaw)+np.sin(yaw)*np.cos(r) , -np.sin(p)*np.sin(r)*np.sin(yaw) + np.cos(r)*np.cos(yaw), -np.sin(r)*np.cos(p), y],
                  [-np.sin(p)*np.cos(r)*np.cos(yaw)+np.sin(r)*np.sin(yaw), np.sin(p)*np.sin(yaw)*np.cos(r)+np.sin(r)*np.cos(yaw)   , np.cos(p)*np.cos(r) , z],
                  [0                                                     , 0                                                       , 0                   , 1]])
    return t0i



def quadcopter_dynamics(w,vx,vy,vz,r,p,yaw,vr,vp,vyaw,agi):
    w1=w[0][0]
    w2=w[1][0]
    w3=w[2][0]
    w4=w[3][0]
    T1 = k * (w1 * w1 + w2 * w2 + w3 * w3 + w4 * w4)  # total thrust force(N) in direction z, w is the angular speed of the propeller
    T2 = k * (w1 * w1 + w4 * w4 - w2 * w2 - w3 * w3) * l * np.cos(alpha)  # total roll torque(Nm)
    T3 = k * (w1 * w1 + w2 * w2 - w3 * w3 - w4 * w4) * l * np.sin(alpha)  # total pitch torque(Nm)
    T4 = b * (w1 * w1 + w3 * w3 - w2 * w2 - w4 * w4)  # total yaw torque(Nm), b is the torque coefficient
    omega = w1 + w3 - w2 - w4
    ax = - Ax / m * vx+agi[0][0]  # translational acceleration in x direction of the moving body frame i
    ay = - Ay / m * vy+agi[1][0]  # translational acceleration in y direction of the moving body frame i
    az = T1 / m - Az / m * vz+agi[2][0] # translational acceleration in z direction of the moving body frame i
    ati=np.array([[ax],[ay],[az]])
    vx = vx + Ts * ax
    vy = vy + Ts * ay
    vz = vz + Ts * az
    dtra = np.array([[Ts*vx-1/2*Ts*Ts*ax],
                     [Ts*vy-1/2*Ts*Ts*ay],
                     [Ts*vz-1/2*Ts*Ts*az]])
    vt = np.array([[vx], [vy], [vz]])
    aroll = (Iyy - Izz) / Ixx * vp * vyaw - Ir / Ixx * vp * omega + T2 / Ixx - Ar / Ixx * vr # rotational acceleration in x direction of the moving body frame i
    apitch = (Izz - Ixx) / Iyy * vr * vyaw + Ir / Iyy * vr * omega + T3 / Iyy - Ar / Iyy * vp # rotational acceleration in y direction of the moving body frame i
    ayaw = (Ixx - Iyy) / Izz * vr * vp + T4 / Izz - Ar / Izz * vyaw # rotational acceleration in z direction of the moving body frame i
    aai=np.array([[aroll],[apitch],[ayaw]])
    vr = vr + Ts * aroll
    vp = vp + Ts * apitch
    vyaw = vyaw + Ts * ayaw
    dang = np.array([[Ts * vr - 1 / 2 * Ts * Ts * aroll],
                     [Ts * vp - 1 / 2 * Ts * Ts * apitch],
                     [Ts * vyaw - 1 / 2 * Ts * Ts * ayaw]])
    va = np.array([[vr], [vp], [vyaw]])
    return vt,va,ati,aai,dtra,dang

def draw_drone(ax,p0,p0l,p0u,p10,p20,p30,p40,p110,p210,p310,p410,p120,p220,p320,p420):
    #ax.scatter3D(p0[0][0], p0[1][0], 5)
    ax.scatter3D(p0[0][0], p0[1][0], p0[2][0])
    ax.scatter3D(p0l[0][0], p0l[1][0], p0l[2][0])
    ax.scatter3D(p0u[0][0], p0u[1][0], p0u[2][0])
    ax.plot3D([p10[0][0], p0u[0][0]], [p10[1][0], p0u[1][0]], [p10[2][0], p0u[2][0]])
    ax.plot3D([p20[0][0], p0u[0][0]], [p20[1][0], p0u[1][0]], [p20[2][0], p0u[2][0]])
    ax.plot3D([p30[0][0], p0u[0][0]], [p30[1][0], p0u[1][0]], [p30[2][0], p0u[2][0]])
    ax.plot3D([p40[0][0], p0u[0][0]], [p40[1][0], p0u[1][0]], [p40[2][0], p0u[2][0]])
    ax.plot3D([p110[0][0], p210[0][0]], [p110[1][0], p210[1][0]], [p110[2][0], p210[2][0]])
    ax.plot3D([p210[0][0], p310[0][0]], [p210[1][0], p310[1][0]], [p210[2][0], p310[2][0]])
    ax.plot3D([p310[0][0], p410[0][0]], [p310[1][0], p410[1][0]], [p310[2][0], p410[2][0]])
    ax.plot3D([p410[0][0], p110[0][0]], [p410[1][0], p110[1][0]], [p410[2][0], p110[2][0]])
    ax.plot3D([p120[0][0], p220[0][0]], [p120[1][0], p220[1][0]], [p120[2][0], p220[2][0]])
    ax.plot3D([p220[0][0], p320[0][0]], [p220[1][0], p320[1][0]], [p220[2][0], p320[2][0]])
    ax.plot3D([p320[0][0], p420[0][0]], [p320[1][0], p420[1][0]], [p320[2][0], p420[2][0]])
    ax.plot3D([p420[0][0], p120[0][0]], [p420[1][0], p120[1][0]], [p420[2][0], p120[2][0]])
    ax.plot3D([p110[0][0], p120[0][0]], [p110[1][0], p120[1][0]], [p110[2][0], p120[2][0]])
    ax.plot3D([p210[0][0], p220[0][0]], [p210[1][0], p220[1][0]], [p210[2][0], p220[2][0]])
    ax.plot3D([p310[0][0], p320[0][0]], [p310[1][0], p320[1][0]], [p310[2][0], p320[2][0]])
    ax.plot3D([p410[0][0], p420[0][0]], [p410[1][0], p420[1][0]], [p410[2][0], p420[2][0]])


def input_update(w,rv,dv,e_pre,e_integral_pre,rv1,dv1,e_pre1,e_integral_pre1):
    cof1=10.0
    cof2=1.0
    cof3=100.0
    cof11 = 200.0
    cof21 = 40.0 # reduce the stead state error, but increase the oscillation
    cof31 = 200.0 # reduce the oscillation
    e=dv-rv
    e_integral=e_integral_pre+e
    e_derivative=(e-e_pre)/Ts
    e1 = dv1 - rv1
    e_integral1 = e_integral_pre1 + e1
    e_derivative1 = (e1 - e_pre1) / Ts
    w[0][0] = np.clip(+(cof1)*e+cof2*e_integral+cof3*e_derivative+(cof11)*e1+cof21*e_integral1+cof31*e_derivative1,0,500)
    w[1][0] = np.clip(00+(cof1+0.3)*e+(cof2)*e_integral+(cof3)*e_derivative+(cof11)*e1+cof21*e_integral1+cof31*e_derivative1,0,500)
    w[2][0] = np.clip(00+(cof1+0.3)*e+(cof2)*e_integral+(cof3)*e_derivative+(cof11)*e1+cof21*e_integral1+cof31*e_derivative1,0,500)
    w[3][0] = np.clip(+(cof1)*e+cof2*e_integral+cof3*e_derivative+cof11*e1+(cof21)*e_integral1+cof31*e_derivative1,0,500)
    return w,e,e_integral,e1,e_integral1



def claculate_angular_position(T0it):
    if T0it[0][2] != 1 and T0it[0][2] != -1:
        proll = np.atan2(-T0it[1][2] , T0it[2][2])
        ppitch = np.arcsin(T0it[0][2])
        pyaw = np.atan2(-T0it[0][1] ,T0it[0][0])
    elif T0it[0][2] == 1:
        proll =0
        ppitch = 90 / 180 * np.pi
        pyaw = np.atan2(T0it[1][0],T0it[1][1])-proll
    else:
        proll = 0
        ppitch = -90 / 180 * np.pi
        pyaw = proll-np.atan2(-T0it[1][0],T0it[1][1])
    return proll,ppitch,pyaw


if __name__=="__main__":
    pi=np.array([[0.0],[0.0],[0.0],[1]]) # c.o.g. of the quadcopter initial position in the body frame
    piu=np.array([[0.0],[0.0],[h/2],[1]]) # upper point of the quadcopter initial position in the body frame
    pil = np.array([[0.0], [0.0], [-h / 2], [1]]) # lower point of the quadcopter initial position in the body frame
    vti=np.array([[0.0],[0.0],[0.0]]) # initial translational velocity in the body frame
    ati=np.array([[0.0],[0.0],[0.0]]) # initial translational acceleration in the body frame
    vai = np.array([[0.0], [0.0], [0.0]]) # initial rotational velocity in the body frame
    aai = np.array([[0.0], [0.0], [0.0]]) # initial rotational acceleration in the body frame
    dtra = Ts*vti+1/2*Ts*Ts*ati  # initial translational displacements around body axis
    dang = Ts*vai+1/2*Ts*Ts*aai  # initial rotational displacements around body axis
    dang_pre=np.array([[0.0],[0.0],[0.0]])
    ag = np.array([[0], [0], [-9.81]])  # gravitational acceleration in the fixed frame(m/s^2)
    p1i=np.array([[-l*np.sin(alpha)],[l*np.cos(alpha)],[h/2],[1]]) # arm 1 end position in the body frame
    p2i = np.array([[-l * np.sin(alpha)], [-l * np.cos(alpha)], [h/2], [1]]) # arm 2 end position in the body frame
    p3i = np.array([[l * np.sin(alpha)], [-l * np.cos(alpha)], [h/2], [1]]) # arm 3 end position in the body frame
    p4i = np.array([[l * np.sin(alpha)], [l * np.cos(alpha)], [h/2], [1]]) # arm 4 end position in the body frame
    p11i = np.array([[-l/4 * np.sin(alpha)], [l/4 * np.cos(alpha)], [h / 2], [1]])
    p21i = np.array([[-l/4 * np.sin(alpha)], [-l/4 * np.cos(alpha)], [h / 2], [1]])
    p31i = np.array([[l/4 * np.sin(alpha)], [-l/4 * np.cos(alpha)], [h / 2], [1]])
    p41i = np.array([[l/4 * np.sin(alpha)], [l/4 * np.cos(alpha)], [h / 2], [1]])
    p12i = np.array(
        [[-l / 4 * np.sin(alpha)], [l / 4 * np.cos(alpha)], [-h / 2], [1]])  # arm 1 end position in the body frame
    p22i = np.array(
        [[-l / 4 * np.sin(alpha)], [-l / 4 * np.cos(alpha)], [-h / 2], [1]])  # arm 2 end position in the body frame
    p32i = np.array(
        [[l / 4 * np.sin(alpha)], [-l / 4 * np.cos(alpha)], [-h / 2], [1]])  # arm 3 end position in the body frame
    p42i = np.array(
        [[l / 4 * np.sin(alpha)], [l / 4 * np.cos(alpha)], [-h / 2], [1]])  # arm 4 end position in the body frame
    T0it = transformation_matrix0i(0.0, 0.0, h/2, 0.0, 0.0, 0.0)
    T0it=T0it@transformation_matrix0i(dtra[0][0],dtra[1][0],dtra[2][0],dang[0][0],dang[1][0],dang[2][0])
    proll, ppitch, pyaw = claculate_angular_position(T0it)
    Ri0 = np.linalg.inv(rotation_matrix0i(dang[0][0], dang[1][0], dang_pre[0][0], dang_pre[1][0], dang_pre[2][0])) # dont use rotation_matrixi0
    R0it=rotation_matrix0i(dang[0][0], dang[1][0], dang_pre[0][0],dang_pre[1][0],dang_pre[2][0])
    p0=T0it@pi # c.o.g. of the quadcopter initial position in the fixed frame
    path0=p0
    p0u=T0it@piu
    p0l=T0it@pil
    p10=T0it@p1i
    p20 =T0it @ p2i
    p30 =T0it @ p3i
    p40 = T0it @ p4i
    p110 = T0it @ p11i
    p210 = T0it @ p21i
    p310 = T0it @ p31i
    p410 = T0it @ p41i
    p120 = T0it @ p12i
    p220 = T0it @ p22i
    p320 = T0it @ p32i
    p420 = T0it @ p42i
    vti = Ri0 @ vti
    vai = Ri0 @ vai
    vt0 = R0it @ vti
    va0 = R0it @ vai
    at0 = R0it @ ati
    aa0 = R0it @ aai
    agi = Ri0 @ ag # gravitational acceleration in the body frame(m/s^2)
    w=np.array([[0.0],[0.0],[0.0],[0.0]])
    timestep=0
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    draw_drone(ax, p0, p0l, p0u, p10, p20, p30, p40, p110, p210, p310, p410, p120, p220, p320, p420)
    ax.view_init(elev=10, azim=50, roll=0)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    plt.axis('equal')
    plt.title(
        f'Initial Quadcopter state|Tra.Accel(XYZ): {at0[0][0]:.3f}m/s^2  |{at0[1][0]:.3f}m/s^2  |{at0[2][0]:.3f}m/s^2\n'
                      f'                |Ang.Accel(XYZ): {aa0[0][0]:.3f}rad/s^2|{aa0[1][0]:.3f}rad/s^2|{aa0[2][0]:.3f}rad/s^2\n'
                      f'                |Ang.Posit(XYZ): {proll:.3f}rad        |{ppitch:.3f}rad       |{pyaw:.3f}rad\n'
                      f'                |Tra.posit(XYZ): {p0[0][0]:.3f}m       |{p0[1][0]:.3f}m       |{p0[2][0]:.3f}m',
        y=0.9)
    plt.pause(5)
    e_cur =0
    e_integral_cur=0
    e_cur1 = 0
    e_integral_cur1 = 0
    while True:
        dang_pre=dang
        #print(T0it)
        vti,vai,ati,aai,dtra,dang=quadcopter_dynamics(w,vti[0][0],vti[1][0],vti[2][0],dang[0][0],dang[1][0],dang[2][0],vai[0][0],vai[1][0],vai[2][0],agi)
        T0it = T0it@transformation_matrix0i(dtra[0][0], dtra[1][0], dtra[2][0], dang[0][0], dang[1][0], dang[2][0])
        proll, ppitch, pyaw = claculate_angular_position(T0it)
        Ri0 = np.linalg.inv(rotation_matrix0i(dang[0][0], dang[1][0], dang_pre[0][0], dang_pre[1][0], dang_pre[2][0]))
        R0it = R0it@rotation_matrix0i(dang[0][0], dang[1][0], dang_pre[0][0], dang_pre[1][0], dang_pre[2][0])
        p0 = T0it @ pi
        #pathp0=np.concatenate((pathp0,p0),axis=1)
        p0u=T0it@piu
        p0l = T0it @ pil
        p10 = T0it @ p1i
        p20 = T0it @ p2i
        p30 = T0it @ p3i
        p40 = T0it @ p4i
        p110 = T0it @ p11i
        p210 = T0it @ p21i
        p310 = T0it @ p31i
        p410 = T0it @ p41i
        p120 = T0it @ p12i
        p220 = T0it @ p22i
        p320 = T0it @ p32i
        p420 = T0it @ p42i
        vti = Ri0 @ vti
        vai = Ri0 @ vai
        vt0=R0it @ vti
        va0=R0it @ vai
        at0 = R0it @ ati
        aa0 = R0it @ aai
        agi = Ri0 @ agi
        timestep=timestep+1
        w,e_cur,e_integral_cur,e_cur1,e_integral_cur1=input_update(w,vt0[1][0],2.0,e_cur,e_integral_cur,p0[2][0],5,e_cur1,e_integral_cur1)
        ax.cla()
        draw_drone(ax,p0,p0l,p0u,p10,p20,p30,p40,p110,p210,p310,p410,p120,p220,p320,p420)
        ax.view_init(elev=10, azim=50,roll=0)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        plt.axis('equal')
        plt.title(
      f'Quadcopter state|Tra.Accel(XYZ): {at0[0][0]:.3f}m/s^2  |{at0[1][0]:.3f}m/s^2  |{at0[2][0]:.3f}m/s^2\n'
            f'                |Ang.Accel(XYZ): {aa0[0][0]:.3f}rad/s^2|{aa0[1][0]:.3f}rad/s^2|{aa0[2][0]:.3f}rad/s^2\n'
            f'                |Tra.Veloc(XYZ): {vt0[0][0]:.3f}m/s    |{vt0[1][0]:.3f}m/s    |{vt0[2][0]:.3f}m/s\n'
            f'                |Ang.Veloc(XYZ): {va0[0][0]:.3f}rad/s  |{va0[1][0]:.3f}rad/s  |{va0[2][0]:.3f}rad/s\n'
            f'                |Ang.Posit(XYZ): {proll:.3f}rad       |{ppitch:.3f}rad       |{pyaw:.3f}rad\n'
            f'                |Tra.posit(XYZ): {p0[0][0]:.3f}m       |{p0[1][0]:.3f}m       |{p0[2][0]:.3f}m',
        y=0.9)


        plt.pause(0.1)





