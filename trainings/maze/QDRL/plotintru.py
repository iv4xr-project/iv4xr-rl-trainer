from therenv.se_star.intrusion.utils import make_env as make_intrusion_env
from matplotlib import pyplot as plt
from matplotlib import animation

done_gv = False
state_gv = []


def plot_ac(args, actors, device):
    fig = plt.figure()
    fig.set_dpi(100)
    fig.set_size_inches(7, 6.5)
    number_of_actors = len(actors)

    ax = plt.axes(xlim=(-100, 100), ylim=(-100, 100))
    global done_gv
    done_gv = [False] * number_of_actors
    envs = [make_intrusion_env(args) for _ in range(number_of_actors)]
    global state_gv
    state_gv = [envi.reset() for envi in envs]
    goal = plt.Circle((0, 0), 5, fc='r')
    obstacles = []
    for i, obstacle in enumerate(envs[0].space.obstacle_list):
        obstacles.append(plt.Rectangle((obstacle.center.x - obstacle.dimensions.x / 2,
                                        obstacle.center.y - obstacle.dimensions.y / 2
                                        ), obstacle.dimensions.x, obstacle.dimensions.y, fc='g'))

    intruders = [None] * number_of_actors
    for id_env, envi in enumerate(envs):
        for i, intruder_id in envi.action_intruder.items():
            entity_obj = envi.space.get_pedestrian(intruder_id)
            if entity_obj is not None:
                intruders[id_env] = plt.Circle((entity_obj.geometry.position.x,
                                                entity_obj.geometry.position.y
                                                ), 1, fc='C' + str(id_env))

    def init():
        for obs in obstacles:
            ax.add_patch(obs)
        for intruder in intruders:
            ax.add_patch(intruder)
        ax.add_patch(goal)
        return [goal]+obstacles + intruders

    def animate(i):
        global done_gv
        global state_gv

        for id_env, envi in enumerate(envs):
            if not done_gv[id_env]:
                action = actors[id_env](state_gv[id_env])
                next_state, reward, done_, info_ = envi.step(action)
                state_gv[id_env] = next_state
                #   print('done', done_, next_state[0],next_state[1],math.sqrt(((next_state[0]-.5)*100)**2+((next_state[1]-.5)*100)**2))

                done_gv[id_env] = done_
                for i, intruder_id_ in envi.action_intruder.items():
                    entity_obj_ = envi.space.get_pedestrian(intruder_id_)
                    if entity_obj_ is not None:
                        intruders[id_env].center = (entity_obj_.geometry.position.x, entity_obj_.geometry.position.y)
                    else:
                        intruders[id_env].center = (.5, .5)
        return [goal]+obstacles + intruders

    anim = animation.FuncAnimation(fig, animate,
                                   init_func=init,
                                   frames=360,
                                   interval=20,
                                   blit=True)
    plt.show()
    [envo.close() for envo in envs]


def plot_rol(env, rollouts, device):
    import time
    fig = plt.figure()
    fig.set_dpi(100)
    fig.set_size_inches(7, 6.5)
    number_of_actors = len(rollouts)
    ax = plt.axes(xlim=(-100, 100), ylim=(-100, 100))
    goal = plt.Circle((0, 0), 5, fc='r')
    obstacles = []
    for i, obstacle in enumerate(env.space.obstacle_list):
        obstacles.append(plt.Rectangle((obstacle.center.x - obstacle.dimensions.x / 2,
                                        obstacle.center.y - obstacle.dimensions.y / 2
                                        ), obstacle.dimensions.x, obstacle.dimensions.y, fc='g'))

    intruders = [None] * number_of_actors
    for id_rol in range(len(rollouts)):
                intruders[id_rol] = plt.Circle(((rollouts[id_rol][0][0]-.5)*200,
                                                (rollouts[id_rol][0][1]-.5)*200,
                                                ), 3, fc='C' + str(id_rol))

    def init():
        for obs in obstacles:
            ax.add_patch(obs)
        for intruder in intruders:
            ax.add_patch(intruder)
        ax.add_patch(goal)
        return [goal]+obstacles + intruders

    def animate(i):
        time.sleep(.2)

        i= i%200
        for id_rol in range(len(rollouts)):
             if i<len(rollouts[id_rol]):
                 intruders[id_rol].center = ((rollouts[id_rol][i][0]-.5)*200, (rollouts[id_rol][i][1]-.5)*200)
                 #   else:
                 #       intruders[id_rol].center = (.5, .5)
        return [goal]+obstacles + intruders

    anim = animation.FuncAnimation(fig, animate,
                                   init_func=init,
                                   frames=360,
                                   interval=20,
                                   blit=True)
    plt.show()
