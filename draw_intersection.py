from matplotlib import pyplot as plt
from descartes import PolygonPatch
from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon

polygon = Polygon([(0, 0), (1, 1), (1, 0)])
polygon1 = Polygon([(0.5, 0.5), (1.5, 1.5), (1.5, 0.5)])
k = polygon1.intersection(polygon)
x, y = k.exterior.xy
x0, y0 = polygon.exterior.xy
x1, y1 = polygon1.exterior.xy

fig = plt.figure(1, figsize=(5,5), dpi=90)
ax = fig.add_subplot(111)
ax.plot(x, y, color='#6699cc', alpha=0.7,
    linewidth=3, solid_capstyle='round', zorder=2)
ax.plot(x0, y0, color='red', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
ax.plot(x1, y1, color='black', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
ax.set_title('Polygon')
plt.show()
