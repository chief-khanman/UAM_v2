
def static_plot(sim_obj, ax, gpd):
    sim_obj.airspace.location_utm_gdf.plot(ax=ax, color='gray', linewidth=0.6)
    sim_obj.airspace.location_utm_hospital_buffer.plot(ax=ax, color='green', alpha=0.3)
    sim_obj.airspace.location_utm_hospital.plot(ax=ax, color='black')
    #adding vertiports to static plot
    gpd.GeoSeries(sim_obj.sim_vertiports_point_array).plot(ax=ax, color='black')