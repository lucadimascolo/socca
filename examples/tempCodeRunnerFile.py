

# bar_radial = socca.models.Sersic()
# bar_radial.Ie = 5e+0
# bar_radial.re = 2e-2
# bar_radial.ns = 0.5

# print(bar_radial.id)

# bar_geom = socca.models.BarGeometry()
# bar_geom.xc = img.hdu.header['CRVAL1']
# bar_geom.yc = img.hdu.header['CRVAL2']
# bar_geom.e = 0.5
# # bar_geom.inc = socca.priors.boundto(disk_vertical, "inc")
# bar_geom.inc = 0.00
# # bar_geom.theta = socca.priors.boundto(disk_radial, "theta")
# bar_geom.theta = 0.00
# bar_geom.rot = 0.00  # Additional Position Angle Bar

# print(bar_geom.id)

# bar = socca.models.Bar(radial=bar_radial, geometry=bar_geom)

# print(bar.id)