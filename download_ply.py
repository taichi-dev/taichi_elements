import urllib.request

root_url = "https://github.com/taichi-dev/taichi_elements_blender_examples/releases/download/ply/"
# hard code and download

urls = ["bunny_low.ply", "quantized.ply", "simulation.ply", "taichi.ply", "suzanne.npy"]

for url in urls:
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(root_url + url, url)
print("Download finished !")
