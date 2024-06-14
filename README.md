# IR-imaging-2024
Repository for the semester project carried out by Octavio Profeta in the spring of 2024, under the supervision of Pr. Francesco Mondada and Cyril Monette

## training images for the ilastik model

hive2_rpi1_240423-034701Z
hive2_rpi2_240425-164102Z
hive2_rpi3_240423-020502Z
hive2_rpi4_240424-014501Z

# Marche à suivre

All the necessary dataset is in the *live_bees* folder

If you don't have all annotation (live_bees/rpi/masks/.csv):

0. Run the *napari_annotation/pipeline.ipynb* notebook until the indicated mark to start annotating. You need to do that for each missing RPi's masks
    - Once done with one RPi, run the remaining cells to save the masks
    - Rerun the notebook for each missing RPi's masks

If you have all the masks:

1. Run _napari_annotation/csv_to_mask.ipynb_ notebook for each RPi
2. Run _napari_annotation/csv_to_contour.ipynb_ notebook for each RPi

You can now start to run the mask finding pipelines:

3. Run the *aa_thresholding/thresholding.ipynb* notebook for each RPi. The resulting masks are in *a_processed_images/thresholding* folder
4. Run the *aa_region_growing/region_growing.ipynb* notebook for each RPi. The resulting masks are in *a_processed_images/region_growing* folder
5. Run the *aa_ilastik/ilastik.ipynb* notebook for each RPi. The resulting masks are in *a_processed_images/ilastik* folder
6. Run the *aa_optical_flow/optical_flow.ipynb* notebook for each RPi. The resulting masks are in *a_processed_images/optical_flow* folder

Once you have all processed images, run the mask finding pipeline:

7. Run the *contour_finding.ipynb* notebook notebook for each RPi and each method. The resulting masks are in *a_found_masks/method/rpi* folder. 
    - Note: if you did not uncomment the commented code, you can run the next section

8. You can now run the *a_xor_results/xor.ipynb* notebook in order to get the XOR masks. The resulting masks are in *a_xor_results/method* folder. 