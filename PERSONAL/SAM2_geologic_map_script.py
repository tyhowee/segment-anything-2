#%% INSTALL DEPENDENCIES-------------------------------------------------------------------------
import sys
import subprocess
import os
import platform


# Set environment variables for GDAL
def set_gdal_env_vars():
    # Check if GDAL_DATA and PATH are already set
    if "GDAL_DATA" not in os.environ or not os.environ["GDAL_DATA"]:
        # Determine the base path for Anaconda or Miniconda
        conda_base = os.environ.get("CONDA_PREFIX", os.path.dirname(sys.executable))
        gdal_data_path = os.path.join(conda_base, "Library", "share", "gdal")
        
        if os.path.exists(gdal_data_path):
            os.environ["GDAL_DATA"] = gdal_data_path
            print("Set GDAL_DATA to", os.environ["GDAL_DATA"])
        else:
            print("Could not find the GDAL data path. Make sure GDAL is installed properly.")

    if "condabin" not in os.environ["PATH"]:
        conda_bin_path = os.path.join(os.path.dirname(conda_base), "condabin")
        os.environ["PATH"] += os.pathsep + conda_bin_path
        print("Added", conda_bin_path, "to PATH")
    else:
        print("PATH already includes condabin")

set_gdal_env_vars()


def install_and_import(package, import_name=None, use_conda=False):
    """
    Install the package if it's not already installed, and then import it.
    
    Args:
        package (str): The package name to be installed and imported.
        import_name (str): The actual module name to import if different from the package name.
        use_conda (bool): If True, install the package via Conda instead of pip.
    """
    try:
        # Use the provided import name or fallback to the package name
        if import_name:
            __import__(import_name)
        else:
            __import__(package)
    except ImportError:
        print(f"Package '{package}' not found. Installing...")
        if use_conda:
            subprocess.check_call(['conda', 'install', '-c', 'conda-forge', package, '-y'])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"'{package}' installed successfully.")
        # Import again after installation
        globals()[import_name or package] = __import__(import_name or package)
    except Exception as e:
        print(f"An error occurred: {e}")

# List of packages needed for the SAM2 script
required_packages = {
    "numpy": "numpy",
    "tkinter": None,  # tkinter is usually pre-installed with Python
    "torch": "torch",
    "matplotlib": "matplotlib",
    "Pillow": "PIL",  # The actual import name for the Pillow library is PIL
    "rasterio": "rasterio",
    "gdal": "osgeo.gdal",
    "ogr": "osgeo.ogr"
}

# Check and install packages
for package, import_name in required_packages.items():
    if package in ['gdal', 'ogr']:
        # Use Conda for GDAL and its related packages
        install_and_import(package, import_name, use_conda=True)
    else:
        install_and_import(package, import_name)

# Check and install a Qt binding for matplotlib's QtAgg backend
try:
    import PyQt5  # You can choose PyQt6, PySide2, or PySide6 as needed
except ImportError:
    print("Required Qt binding for matplotlib not found. Installing PyQt5...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyQt5"])
    print("PyQt5 installed successfully.")
    globals()["PyQt5"] = __import__("PyQt5")



#%%IMPORT LIBRARIES------------------------------------------------------------------------------------------------------------------------------------

import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
from tkinter import Tk, filedialog
import torch
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
from PIL import Image
import rasterio
from rasterio.transform import from_origin
from osgeo import gdal
from osgeo import ogr

#%%SELECT COMPUTATION DEVICE

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


#%% DEFINE VIS FUNCTIONS

np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


#%% LOAD/PREVIEW IMAGE

# Hide the main tkinter window
Tk().withdraw()

# Open a file dialog to select the image
file_path = filedialog.askopenfilename(title="Select an image file")

# Load and convert the image
image = Image.open(file_path)
image = np.array(image.convert("RGB"))

# Prompt the user to select the output raster file location
output_file_path = filedialog.asksaveasfilename(
    defaultextension=".tif",
    filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")],
    title="Select output file location"
)


#%% LOAD SAM-2

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

predictor.set_image(image)

#%% SELECT POINTS


number_points = 0  # Initialize the number of points

points = []
labels = []

def onclick(event):
    global number_points  # Use global to update the variable
    if event.xdata is None or event.ydata is None:  # Ignore clicks outside the plot
        return
    if event.button == 1:  # Left-click to add a point
        points.append((event.xdata, event.ydata))
        labels.append(1)  # Always use 1 to indicate a new object
        plt.plot(event.xdata, event.ydata, 'go')
        plt.draw()
        number_points += 1  # Increment the point count
    elif event.button == 3 and points:  # Right-click to remove the last point
        points.pop()
        labels.pop()
        plt.clf()
        plt.imshow(image)
        plt.title('Click to select points, right-click to undo. Press Enter to finish.')
        plt.axis('on')
        for p in points:
            plt.plot(p[0], p[1], 'go')
        plt.draw()
        number_points -= 1  # Decrement the point count

def on_key(event):
    if event.key == 'enter':
        plt.close()  # Close the interactive window

# Display the image
fig, ax = plt.subplots()
ax.imshow(image)
plt.title('Click to select points, right-click to undo. Press Enter to finish.')
plt.axis('on')

# Connect the click and key press events
cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
cid_key = fig.canvas.mpl_connect('key_press_event', on_key)

# Wait for the user to finish
plt.show()

# Convert points and labels to NumPy arrays
input_points = np.array(points).reshape(number_points, 1, 2)  # Reshape to Bx1x2 format
input_labels = np.array(labels).reshape(number_points, 1)     # Reshape to Bx1 format

#%% SHOW POINT(S) - DISABLED

#plt.figure(figsize=(10, 10))
#plt.imshow(image)
#show_points(input_points, input_labels, plt.gca())
#plt.axis('on')
#plt.show()  

#print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)

#%% PROCESS MULTIPLE SEPARATE POINTS

masks, scores, _ = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=False,
)

image_points = input_points
print(image_points.shape) # Bx1x2 where B corresponds to number of objects 
image_labels = np.ones((number_points, 1), dtype=int)  # Shape (number_points, 1)
print(image_labels.shape) # Bx1 where B corresponds to number of objects 
image_masks = masks
print(image_masks.shape)  # (batch_size) x (num_predicted_masks_per_input) x H x W


#%% DISPLAY MASKS
plt.figure(figsize=(10, 10))
plt.title("Mask Output - close window to continue")
plt.imshow(image)

# Iterate over each set of points, labels, and masks
for points, labels, masks_per_point in zip(image_points, image_labels, image_masks):
    for mask in masks_per_point:  # Loop over masks for the current point
        show_mask(mask, plt.gca(), random_color=True)
    show_points(points, labels, plt.gca())  # Plot points for the current batch

plt.axis('off')
plt.show()  # Display the plot after all masks and points are plotted


#%% EXPORT MASKS

normalized_masks = (image_masks * 255).astype(np.uint8)  # Convert binary masks to 8-bit format

# Define the dimensions and metadata for the output raster
height, width = image.shape[:2]
transform = from_origin(0, 0, 1, 1)  # Adjust to match the georeferencing

# Create a grayscale output array for the masks
grayscale_masks = np.zeros((height, width), dtype=np.uint8)

# Assign unique grayscale values to each mask (values from 1 to number_points)
for i in range(number_points):
    grayscale_value = int((i + 1) * (255 / (number_points + 1)))  # Calculate unique grayscale value
    grayscale_masks[normalized_masks[i, 0] > 0] = grayscale_value

# Write the grayscale masks to the output file
with rasterio.open(output_file_path, 'w', driver='GTiff', height=height, width=width,
                   count=1, dtype='uint8', transform=transform) as dst:
    dst.write(grayscale_masks, 1)

#%% POLYGONIZE

os.environ['GDAL_DATA'] = r"C:\Users\TyHow\anaconda3\envs\conda_env\Library\share\gdal"


# Write the grayscale masks to the output file
with rasterio.open(output_file_path, 'w', driver='GTiff', height=height, width=width,
                   count=1, dtype='uint8', transform=transform) as dst:
    dst.write(grayscale_masks, 1)

# Extract the directory from the output raster file path
output_dir = os.path.dirname(output_file_path)

# Define the output vector file path in the same directory
output_vector_path = os.path.join(output_dir, "output_vector.shp")

# Run GDAL Polygonize
src_ds = gdal.Open(output_file_path)
src_band = src_ds.GetRasterBand(1)

# Create the output vector file
driver = ogr.GetDriverByName("ESRI Shapefile")
out_ds = driver.CreateDataSource(output_vector_path)
out_layer = out_ds.CreateLayer("polygonized", srs=None)

# Add an attribute to the output layer
field = ogr.FieldDefn("DN", ogr.OFTInteger)
out_layer.CreateField(field)

# Polygonize
gdal.Polygonize(src_band, None, out_layer, 0, [], callback=None)

# Clean up
out_ds = None
src_ds = None


# %%
