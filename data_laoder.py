import pandas as pd
import phys
import tools

dt = pd.read_csv("solar_system_dataset.csv")

dt["Planet"] = dt["Planet"].astype("string").apply(lambda x: x.strip())
dt.set_index("Planet", inplace=True)
print(dt.head())
print(dt.loc['Mercury'])
dt.info()


#  TODO: Fix orbits so that they're not perfect circles
#   (setting the velocity in the direction perp. to distance from the sun was a mistake)

def load_data():
    list_of_objects = {}
    for index, row in dt.iterrows():
        normal = tools.random_normal_vector()
        mass = row['Mass (10^24kg)']*(10**24)
        distance = row['Distance from Sun (10^6 km)']*(10**9)
        pos = normal*distance
        velocityNumber = row['Orbital Velocity (km/s)']*1000
        velocityNormal = tools.getPerpendicularVector(normal)
        velocity = velocityNumber*velocityNormal

        Planet = phys.MyPhysObject(pos[0],pos[1],mass,velocity[0],velocity[1])
        list_of_objects[index]=Planet
        print(index, row['Mass (10^24kg)'], row['Orbital Velocity (km/s)'])
    return list_of_objects
print("----")
load_data()