import math
"""
public Gift(int id, double latitude, double longitude, double weight)
{
    Id = id;
    Weight = weight;

    LatitudeInRads  = (Math.PI/180) * latitude;
    LongitudeInRads = (Math.PI/180) * longitude;

    // Cartesian coordinates, normalized for a sphere of diameter 1.0
    X = 0.5 * Math.Cos(LatitudeInRads) * Math.Sin(LongitudeInRads);
    Y = 0.5 * Math.Cos(LatitudeInRads) * Math.Cos(LongitudeInRads);
    Z = 0.5 * Math.Sin(LatitudeInRads);
}
"""
# public static double HaversineDistance(Gift g1, Gift g2)
#     double dX = g1.X - g2.X;
#     double dY = g1.Y - g2.Y;
#     double dZ = g1.Z - g2.Z;
#
#     double r = Math.Sqrt(dX*dX + dY*dY + dZ*dZ);
#
#     return 2 * AvgEarthRadius * Math.Asin(r);

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
    x1, y1, z1 = point1
    x2, y2, z2 = point2

    # calculate haversine
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    d = math.sqrt(dx * dx + dy * dy + dz * dz)
    return 2 * AVG_EARTH_RADIUS * math.asin(d)
