"""will remove this part after we get haversine package into scripts
"""

from math import radians, cos, sin, asin, sqrt
import pandas as pd
import numpy as np

AVG_EARTH_RADIUS = 6371  # in km


def haversine(point1, point2, miles=False):
    """ Calculate the great-circle distance bewteen two points on the Earth surface.
    :input: two 2-tuples, containing the latitude and longitude of each point
    in decimal degrees.
    Example: haversine((45.7597, 4.8422), (48.8567, 2.3508))
    :output: Returns the distance bewteen the two points.
    The default unit is kilometers. Miles can be returned
    if the ``miles`` parameter is set to True.
    """
    # unpack latitude/longitude
    lat1, lng1 = point1
    lat2, lng2 = point2

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1, lng1, lat2, lng2 = map(radians, (lat1, lng1, lat2, lng2))

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = sin(lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(lng / 2) ** 2
    h = 2 * AVG_EARTH_RADIUS * asin(sqrt(d))
    if miles:
        return h * 0.621371  # in miles
    else:
        return h  # in kilometers

north_pole = (90, 0)
weight_limit = 1000
sleigh_weight = 10


def weighted_trip_length(stops, weights):
    tuples = [tuple(x) for x in stops.values]
    # adding the last trip back to north pole, with just the sleigh weight
    tuples.append(north_pole)
    weights.append(sleigh_weight)

    dist = 0.0
    prev_stop = north_pole
    prev_weight = sum(weights)
    for i in range(len(tuples)):
        dist += haversine(tuples[i], prev_stop) * prev_weight
        prev_stop = tuples[i]
        prev_weight = prev_weight - weights[i]
    return dist


def weighted_reindeer_weariness(all_trips):
    uniq_trips = all_trips.TripId.unique()

    if np.any(all_trips.groupby('TripId').Weight.sum() > weight_limit):
        raise Exception("One of the sleighs over weight limit!")

    dist = 0
    for t in uniq_trips:
        this_trip = all_trips[all_trips.TripId == t]
        dist = dist + weighted_trip_length(this_trip[['Latitude', 'Longitude']], this_trip.Weight.tolist())

    return dist


def grid_cluster(gifts, res):
    grid_lat = np.arange(-90, 90, res)
    grid_lon = np.arange(-180, 180, res)
    gifts = pd.DataFrame(gifts)
    cluster = pd.DataFrame(np.zeros((gifts.shape[0], 2)))
    cluster.index = gifts.index
    cluster.columns = ['cluster_lat', 'cluster_lon']
    gifts = pd.concat([gifts, cluster], axis=1)
    for lat in grid_lat:
        print lat
        for lon in grid_lon:
            lon_eq = ((np.array(gifts['Longitude']) >= lon) * 1 *
                      (np.array(gifts['Longitude']) < lon + res) * 1).astype('bool')
            lon_index = gifts[lon_eq].index
            gifts.at[lon_index, 'cluster_lon'] = lon
            lat_eq = ((np.array(gifts['Latitude']) >= lat) * 1 *
                      (np.array(gifts['Latitude']) < lat + res) * 1).astype('bool')
            lat_index = gifts[lat_eq].index
            gifts.at[lat_index, 'cluster_lat'] = lat
    return gifts

# GiftId   Latitude   Longitude     Weight  cluster_lat  cluster_lon
gifts = pd.read_csv('gifts.csv')
gifts = grid_cluster(gifts, 5)
print gifts.head(n=20)
sample_sub = pd.read_csv('sample_submission.csv')

all_trips = sample_sub.merge(gifts, on='GiftId')

print(weighted_reindeer_weariness(all_trips))
