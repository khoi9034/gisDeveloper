# =====================================================
# Spatial Distribution of Anime Retail Stores in Tokyo
# =====================================================
# This script operationalizes the methodology described in the research proposal:
# “Understanding the Spatial Distribution of Anime Retail Stores in Tokyo, Japan:
#  A Statistical and Cultural Perspective.”
#
# CORE RESEARCH QUESTION:
#   How are anime stores distributed across Tokyo, and to what extent is their
#   location pattern influenced by:
#       1) Population density (primary quantitative factor)
#       2) Economic variables (rent patterns) – future addition
#       3) Accessibility (road network structure) – future addition
#       4) Cultural clustering (e.g., Akihabara, Ikebukuro) – qualitative addition
#
# THIS SCRIPT IMPLEMENTS THE QUANTITATIVE COMPONENT:
#   ✔ Converts municipal population + area → population density raster
#   ✔ Imports anime store point data
#   ✔ Runs:
#         • Nearest Neighbor comparison (Anime vs Random Points)
#         • Raster-to-point conversion for population analysis
#         • Distance-to-population t-test (anime vs random)
#   ✔ Outputs:
#         • GeoPackage storing clean layers
#         • CSV containing statistical comparisons
#         • Layers prepared for later analyses (KDE, Moran’s I)
#
# CONNECTION TO PROPOSAL METHODS:
#   Population Density:
#       - Converted to raster for pixel-level distance analysis.
#       - Used as primary predictor variable.
#
#   Statistical Tests:
#       - Welch’s t-test compares whether anime stores are systematically closer
#         to population centers than random expectation.
#
#   Spatial Point Pattern Framework:
#       - Random points serve as a control model of CSR (Complete Spatial Randomness).
#       - Distance matrices → distribution comparison.
#
#   Future Steps (already scaffolded in this code):
#       - Road network influence (from Han 2019)
#       - Rent pressure analysis (Tsoutsos & Photis 2020)
#       - Cultural hotspot weighting (Kitabayashi & Yamazaki 2018)

# =====================================================


import arcpy
import os
import sys
import numpy as np
import csv
import traceback

# Setting up stats library for t test and other stats test later. 
#put in try so code doesent break in case scipy is not preinstalled
# try:
#     from scipy import stats
#     print("✅ SciPy detected: statistical tests enabled.")
# except ImportError:
#     stats = None
#     print("⚠️ SciPy not found — Welch t-test will be skipped.")


# =====================================================
# STEP 1 — Folder Setup
# =====================================================
print("\n[STEP 1] Setting up project folders...")
arcpy.env.overwriteOutput = True

project_folder = r"C:\ArcPyProjects\AnimeStoreProximityAnalysis"
input_folder = os.path.join(project_folder, "inputs")
output_folder = os.path.join(project_folder, "output")
boundary_fc = os.path.join(input_folder, "CorrectedBoundariesTokyoMunicipals.shp")


if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created output folder: {output_folder}")
else:
    print(f"Output folder found: {output_folder}")

# STEP 1.5 & 1.6 — Prepare Population Data and Rasterize
# =====================================================
print("\n[STEP 1.5/1.6] Preparing population data and rasterization...")

population_tokyo_csv = os.path.join(input_folder, "TokyoPopulation.csv")
boundary_projected = os.path.join(output_folder, "TokyoMunicipalBoundaries_projected.shp")
boundary_joined_w_pop = os.path.join(output_folder, "TokyoMunicipalBoundaries_Joined.shp")
pop_raster = os.path.join(output_folder, "population_density_utm.tif")

# ----------------------------
# Project raw boundary
# ----------------------------
try:
    if arcpy.Exists(boundary_projected):
        arcpy.management.Delete(boundary_projected)
    arcpy.management.Project(
        in_dataset=boundary_fc,
        out_dataset=boundary_projected,
        out_coor_system=arcpy.SpatialReference(6697)  # JGD2011 / UTM Zone 54N
    )
    print("✅ Boundary projected to JGD2011 / UTM Zone 54N")
except Exception as e:
    print("❌ Failed to project boundary:", e)
    raise

# ----------------------------
# Join population CSV
# ----------------------------
try:
    layer_name = "boundary_lyr"
    arcpy.management.MakeFeatureLayer(boundary_projected, layer_name)
    arcpy.management.AddJoin(layer_name, "ADM2_EN", population_tokyo_csv, "ADM2_EN")
    arcpy.management.CopyFeatures(layer_name, boundary_joined_w_pop)
    arcpy.management.RemoveJoin(layer_name)
    print(f"✅ Population CSV joined to boundary: {boundary_joined_w_pop}")
except Exception as e:
    print("❌ Failed to join population CSV:", e)
    raise

# ----------------------------
# Add area_km2 field
# ----------------------------
try:
    fields = [f.name for f in arcpy.ListFields(boundary_joined_w_pop)]
    if "area_km2" not in fields:
        arcpy.management.AddField(boundary_joined_w_pop, "area_km2", "DOUBLE")
        arcpy.management.CalculateField(
            boundary_joined_w_pop,
            "area_km2",
            "!shape.area@SQUAREKILOMETERS!",
            "PYTHON3"
        )
        print("✅ area_km2 field created and calculated.")
except Exception as e:
    print("❌ Failed to add/calculate area_km2:", e)
    raise

# ----------------------------
# Calculate population density
# ----------------------------
try:
    fields = [f.name for f in arcpy.ListFields(boundary_joined_w_pop)]
    if "pop_dens" not in fields:
        arcpy.management.AddField(boundary_joined_w_pop, "pop_dens", "DOUBLE")
    arcpy.management.CalculateField(
        boundary_joined_w_pop,
        "pop_dens",
        "!TokyoPop_1! / !area_km2!",
        "PYTHON3"
    )
    print("🧮 Population density calculated.")
except Exception as e:
    print("❌ Failed to calculate pop_dens:", e)
    raise

# ----------------------------
# Cap extreme population densities
# ----------------------------
try:
    print("Capping extreme population densities at 99th percentile...")
    dens_values = [row[0] for row in arcpy.da.SearchCursor(boundary_joined_w_pop, ["pop_dens"])]
    cap_value = np.percentile(dens_values, 99)
    print(f"99th percentile cap value: {cap_value}")

    with arcpy.da.UpdateCursor(boundary_joined_w_pop, ["pop_dens"]) as cursor:
        for row in cursor:
            if row[0] > cap_value:
                row[0] = cap_value
                cursor.updateRow(row)
    print("✅ Extreme pop_dens values capped.")
except Exception as e:
    print("❌ Failed to cap pop_dens:", e)
    raise

# ----------------------------
# Clean bad polygons (zero/negative area, population, density)
# ----------------------------
try:
    cleanedBoundariesJoined = os.path.join(output_folder, "TokyoMunicipalBoundaries_Joined_Cleaned.shp")
    if arcpy.Exists(cleanedBoundariesJoined):
        arcpy.management.Delete(cleanedBoundariesJoined)
    arcpy.management.CopyFeatures(boundary_joined_w_pop, cleanedBoundariesJoined)

    bad_rows_deleted = 0
    with arcpy.da.UpdateCursor(cleanedBoundariesJoined, ["area_km2", "TokyoPop_1", "pop_dens"]) as cursor:
        for row in cursor:
            area, pop, dens = row
            if area <= 0 or pop <= 0 or dens <= 0:
                cursor.deleteRow()
                bad_rows_deleted += 1
    print(f"✅ Deleted {bad_rows_deleted} bad polygons. Cleaned data ready for rasterization.")
except Exception as e:
    print("❌ Failed to clean bad polygons:", e)
    raise

# ----------------------------
# Rasterize population density
# ----------------------------
try:
    pop_raster = os.path.join(output_folder, "population_density_utm.tif")
    utm54n = arcpy.SpatialReference(6697)  # JGD2011 / UTM Zone 54N

    arcpy.conversion.PolygonToRaster(
        in_features=cleanedBoundariesJoined,
        value_field="pop_dens",
        out_rasterdataset=pop_raster,
        cell_assignment="CELL_CENTER",
        cellsize=0.0016
    )
    arcpy.management.DefineProjection(pop_raster, utm54n)
    arcpy.management.CalculateStatistics(pop_raster)

    print(f"✅ Population raster created: {pop_raster}")

except Exception as e:
    print("❌ Rasterization failed:", e)
    raise












# =====================================================
# STEP 2 — Data Paths
# =====================================================
print("\n[STEP 2] Defining input & output datasets...")

anime_fc = os.path.join(input_folder, "AnimeStoresLocationsWithinBoundary.shp")




results_csv = os.path.join(output_folder, "distribution_results.csv")

gpkg_layers = {
    "anime": "anime_stores",
    "random": "random_points",
    "pop": "population_points",
    "anime_near": "anime_near",
    "random_near": "random_near"
}

# =====================================================
# STEP 3 — ArcPy Environment
# =====================================================
print("\n[STEP 3] Configuring ArcPy environment...")

arcpy.env.workspace = output_folder
print(f"Workspace set to {arcpy.env.workspace}")

# =====================================================
# STEP 4 — Helper Functions
# =====================================================




def check_exists_layer(path, datatype="Feature Class"):
    if not arcpy.Exists(path):
        raise FileNotFoundError(f"❌ Missing {datatype}: {path}")
    print(f"✔ Exists: {path}")

def export_to_gpkg(input_fc, gpkg_path, layer_name):
    check_exists_layer(input_fc)
    print(f"Exporting {layer_name} to GeoPackage...")
    arcpy.conversion.FeatureClassToFeatureClass(
        input_fc, gpkg_path, layer_name
    )







analysis_gdb = os.path.join(output_folder, "analysis.gdb")
if not arcpy.Exists(analysis_gdb):
    arcpy.management.CreateFileGDB(output_folder, "analysis.gdb")


def generate_random_points(n, boundary_fc, gdb, layer_name):
    """
    Generate exactly n random points within a (multi-)polygon boundary.
    Handles multi-polygons by dissolving internally to ensure correct point count.
    """
    import tempfile

    # Ensure n is integer
    n = int(n)
    print(f"Generating {n} random points constrained by {boundary_fc}...")

    # Temporary dissolved polygon (to avoid per-part point multiplication)
    with tempfile.TemporaryDirectory() as tmpdir:
        dissolved_fc = os.path.join(tmpdir, "dissolved_boundary.shp")
        
        # Dissolve polygons to a single multipart polygon
        arcpy.management.Dissolve(boundary_fc, dissolved_fc)
        
        # Delete existing random layer if present
        random_layer_path = os.path.join(gdb, layer_name)
        if arcpy.Exists(random_layer_path):
            arcpy.management.Delete(random_layer_path)

        # Generate points inside the dissolved polygon
        arcpy.management.CreateRandomPoints(
            out_path= gdb,
            out_name=layer_name,
            constraining_feature_class=dissolved_fc,
            number_of_points_or_field=n 
        )
        random_fc_path = os.path.join(gdb, layer_name)
        

        projected_path = os.path.join(gdb, f"{layer_name}_projected")
        arcpy.management.Project(
            in_dataset=random_fc_path,
            out_dataset=projected_path,
            out_coor_system=arcpy.SpatialReference(6697)
        )   

    print(f"✅ Random points generated: {random_layer_path}")
    return projected_path








def population_centroids(boundary_fc, gpkg_path, layer_name):
    """
    Generate one point per municipal polygon at the centroid location.
    Avoids creating millions of points from tiny raster cells.
    """
    
    arcpy.management.FeatureToPoint(boundary_fc, os.path.join(gpkg_path, layer_name), "INSIDE")
    return os.path.join(gpkg_path, layer_name)


def generate_near_table(in_fc, near_fc, out_table): 
    """
    Generate a Near Table between two feature classes.

    ArcPy automatically creates the following fields in the output table:
        - IN_FID: OBJECTID of the feature from the input layer (in_fc)
        - NEAR_FID: OBJECTID of the closest feature in the near layer (near_fc)
        - NEAR_DIST: Distance between IN_FID and NEAR_FID in map units
        - RANK: Rank of the near feature (1 = closest)

    How it works:
        1. For each feature in in_fc (identified by IN_FID):
            - Search near_fc to find the closest feature.
            - Record the OBJECTID of the nearest feature as NEAR_FID.
            - Calculate distance between the two points and store as NEAR_DIST.
        2. If closest="1", only the nearest neighbor is returned (no additional ranks).
        3. The resulting table can be used to draw lines or extract distances for statistics.

    Example table row:
        IN_FID | NEAR_FID | NEAR_DIST
        -------|----------|----------
        1      | 13       | 527

    This means:
        - Input feature 1 is closest to feature 13 in the near layer.
        - Distance between them is 527 units.
    """
    print(f"Generating near table: {out_table}...")
    if arcpy.Exists(out_table):
        arcpy.management.Delete(out_table)

    arcpy.analysis.GenerateNearTable(
        in_fc, near_fc, out_table,
        closest="1", method="GEODESIC"
    )


def near_distances_to_array(near_table):
    """
    Extract distances from a Near Table and return them as a NumPy array.
    
    Parameters:
    - near_table: path to the Near Table (output of arcpy.analysis.GenerateNearTable)
    
    Returns:
    - NumPy array of distances (float) from input features to nearest features
    """
    
    # Initialize an empty list to store distances
    dist = []
    
    # Open a search cursor on the Near Table
    # arcpy.da.SearchCursor iterates over rows in a table or feature class
    # ["NEAR_DIST"] specifies that we only want the NEAR_DIST field (distance in map units)
    with arcpy.da.SearchCursor(near_table, ["NEAR_DIST"]) as cur:
        
        # Loop through each row returned by the cursor
        # Each row is a tuple; (d,) unpacks the single value in the tuple
        for (d,) in cur:
            
            # Check if the distance value is not None (sometimes Near Table can have nulls)
            if d is not None:
                
                # Add the valid distance value to the list
                dist.append(d)
    
    # Convert the Python list to a NumPy array of floats
    # This makes it easier to perform statistics like mean, median, std, etc.
    return np.array(dist, dtype=float)


def near_table_to_lines(near_table, in_fc, near_fc, out_fc):#custom funciton that turns near table into lines
    arcpy.management.CreateFeatureclass(
        out_path=os.path.dirname(out_fc),
        out_name=os.path.basename(out_fc),
        geometry_type="POLYLINE",
        spatial_reference=in_fc 
  
    )
    in_points = {row[0]: row[1] for row in arcpy.da.SearchCursor(in_fc, ["OBJECTID", "SHAPE@XY"])}
    near_points = {row[0]: row[1] for row in arcpy.da.SearchCursor(near_fc, ["OBJECTID", "SHAPE@XY"])}
    
    with arcpy.da.InsertCursor(out_fc, ["SHAPE@"]) as ins_cur:
        with arcpy.da.SearchCursor(near_table, ["IN_FID", "NEAR_FID"]) as near_cur:           
            for in_fid, near_fid in near_cur:
                start = in_points[in_fid]
                end = near_points[near_fid]
                line = arcpy.Polyline(arcpy.Array([arcpy.Point(*start), arcpy.Point(*end)]))
                ins_cur.insertRow([line])




    print(f"✅ Near lines created: {out_fc}")

    



def Moran_LISA_analysis(poly_fc, value_field, output_fc):
    """
    Run LISA analysis using ArcPy OptimizedOutlierAnalysis.
    Print the fields present in the output for verification.
    """
    print("\n🔹 Running LISA analysis using ArcPy OptimizedOutlierAnalysis...")

    # Delete output if it exists
    if arcpy.Exists(output_fc):
        arcpy.management.Delete(output_fc)

    try:
        arcpy.stats.OptimizedOutlierAnalysis(
            Input_Features=poly_fc,
            Analysis_Field=value_field,
            Output_Features=output_fc
        )
        print(f"✅ LISA analysis complete. Output saved at: {output_fc}")

        # Check fields in output
        fields = [f.name for f in arcpy.ListFields(output_fc)]
        print(f"ℹ️ Fields in LISA output: {fields}")

    except Exception as e:
        print(f"❌ LISA analysis failed: {e}")
        fields = [f.name for f in arcpy.ListFields(output_fc)]
        print(f"ℹ️ Fields found in output (partial result if any): {fields}")
        raise

























    

# =====================================================
# STEP 5 — Main Workflow        
# =====================================================
def main():
    
   
    try:
       
        print("\n🚀 Starting Anime Store Spatial Distribution Analysis...\n")
        
        export_to_gpkg(anime_fc, analysis_gdb, gpkg_layers["anime"])
       
       
        n_anime = int(arcpy.management.GetCount(anime_fc).getOutput(0))

        if n_anime == 0:
            raise ValueError("Anime store point layer is empty.")

        print(f"Anime store count: {n_anime}")





        # generate random control points
        random_shp = "RandomPoints"
        projected_random_points= generate_random_points(n_anime, boundary_fc, analysis_gdb, random_shp)
        # Export random points to shapefile so OBJECTID exists cuz neartable to line funciton needs it and i dont wana have to rewrite the functin to work with gpkg layers
       #then i have ot export back as lyaer cuz the rest of my code or specifically the neawrtable to line funcitno uses layers.
        export_to_gpkg(projected_random_points, analysis_gdb, gpkg_layers["random"]) 



        # convert population raster → points
        population_centroids(cleanedBoundariesJoined, analysis_gdb, gpkg_layers["pop"])
        print(f"Population points created")
            
        near_lines_points = os.path.join(output_folder, "near_lines_points.shp")
        # near tables
        generate_near_table(
            os.path.join(analysis_gdb, gpkg_layers["anime"]),
            os.path.join(analysis_gdb, gpkg_layers["pop"]),
            os.path.join(analysis_gdb, gpkg_layers["anime_near"])

        )

        near_table_to_lines(
        os.path.join(analysis_gdb, gpkg_layers["anime_near"]),
        os.path.join(analysis_gdb, gpkg_layers["anime"]),
        os.path.join(analysis_gdb, gpkg_layers["pop"]),
        near_lines_points
        
        )




        near_lines_random = os.path.join(output_folder, "near_lines_random.shp")    
        
        generate_near_table(
            os.path.join(analysis_gdb, gpkg_layers["random"]),
            os.path.join(analysis_gdb, gpkg_layers["pop"]),
            os.path.join(analysis_gdb, gpkg_layers["random_near"]),

        )

        near_table_to_lines(
        os.path.join(analysis_gdb, gpkg_layers["random_near"]),
        os.path.join(analysis_gdb, gpkg_layers["random"]),
        os.path.join(analysis_gdb, gpkg_layers["pop"]),
        near_lines_random
        


        )


        

        # extract distances
        anime_dist = near_distances_to_array(os.path.join(analysis_gdb, gpkg_layers["anime_near"]))
        random_dist = near_distances_to_array(os.path.join(analysis_gdb, gpkg_layers["random_near"]))

        # descriptive statistics
        mean_anime = np.mean(anime_dist)
        mean_random = np.mean(random_dist)
        med_anime = np.median(anime_dist)
        med_random = np.median(random_dist)
        sd_anime = np.std(anime_dist, ddof=1)
        sd_random = np.std(random_dist, ddof=1)

        



        # =====================================================
        # Stats library — Welch t-test
        # =====================================================
        try:
            from scipy import stats
            print("✅ SciPy detected: statistical tests enabled.")
        except ImportError:
            stats = None
            print("⚠️ SciPy not found — Welch t-test will be skipped.")



        
        
        # statistical test
        if stats is not None:
            t, p = stats.ttest_ind(
                anime_dist, random_dist,
                equal_var=False, nan_policy='omit'
            )
        else:
            t, p = None, None

        # output CSV
        with open(results_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "metric", "anime_mean", "random_mean",
                "anime_median", "random_median",
                "anime_std", "random_std",
                "t_stat", "p_value"
            ])
            writer.writerow([
                "distance_to_population",
                mean_anime, mean_random,
                med_anime, med_random,
                sd_anime, sd_random,
                t, p
            ])

        print(f"\n📊 Results saved: {results_csv}")
        print("\n🎉 Analysis complete!")

        # print descriptive statistics and t-test
        print("\n📊 Distance-to-Population Statistics:")
        print(f"Anime stores: mean={mean_anime:.2f}, median={med_anime:.2f}, std={sd_anime:.2f}")
        print(f"Random points: mean={mean_random:.2f}, median={med_random:.2f}, std={sd_random:.2f}")
        if t is not None and p is not None:
            print(f"Welch t-test: t_stat={t:.3f}, p_value={p:.4f}")
        else:
            print("Welch t-test skipped (SciPy not installed).")










        ###################LISA ON POPULAITON
        print("\n==============================")
        print(" Running LISA analysis (Population Density)")
        print("==============================\n")

        # Ensure analysis.gdb exists
        

     
        lisa_output_population = os.path.join(analysis_gdb, "TokyoMunicipals_LISA")

        # Delete existing output if needed
        if arcpy.Exists(lisa_output_population):
            arcpy.management.Delete(lisa_output_population)
            print(f"🗑️ Removed previous population LISA: {lisa_output_population}")
        else:
            print("✅ No previous population LISA found.")

        # Run LISA on polygon population density
        Moran_LISA_analysis(cleanedBoundariesJoined, "pop_dens", lisa_output_population)






        # -------------------------------
        # HOT SPOT ANALYSIS (Gi*) FOR ANIME STORE CLUSTERS
        # -------------------------------
        print("\n==============================")
        print(" Running Getis-Ord Gi* Hot Spot Analysis on Anime Stores")
        print("==============================\n")

        anime_points = os.path.join(analysis_gdb, gpkg_layers["anime"])
        hotspot_output = os.path.join(analysis_gdb, "AnimeStores_GiHotSpot")

        # delete old
        if arcpy.Exists(hotspot_output):
            arcpy.management.Delete(hotspot_output)




        anime_count_fc = os.path.join(analysis_gdb, "TokyoMunicipals_AnimeCount")

        # Delete if exists
        if arcpy.Exists(anime_count_fc):
            arcpy.management.Delete(anime_count_fc)

        arcpy.analysis.SummarizeWithin(
            in_polygons=cleanedBoundariesJoined,
            in_sum_features=anime_points,
            out_feature_class=anime_count_fc,
            sum_fields=[],              # No numeric fields to sum, we just want POINT_COUNT
            group_field="",             # None
            keep_all_polygons=True      # KEEP polygons even if they contain 0 stores
        )

        target_crs = cleanedBoundariesJoined
        # Add COUNT field (arcpy does this automatically as "POINT_COUNT")
        anime_points_projected = os.path.join(analysis_gdb, "AnimeStores_UTM")
        arcpy.management.Project(
            in_dataset=anime_points,
            out_dataset=anime_points_projected,
            out_coor_system=target_crs
        )


        arcpy.stats.HotSpots(
            Input_Feature_Class=anime_count_fc,
            Output_Feature_Class=hotspot_output,
            Input_Field="POINT_COUNT",
            Conceptualization_of_Spatial_Relationships="FIXED_DISTANCE_BAND",
            Distance_Band_or_Threshold_Distance="1500 Meters",
            Standardization="ROW"
        )







        print(f"🔥 Anime Store Hot Spot layer created: {hotspot_output}")






























    except Exception as e:
        print("\n🚨 ERROR OCCURRED!")
        print("Details:", e)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()