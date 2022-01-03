import os


STORAGE_DIR = '../_DATASETS_/occMapDataset/val/_scenes/'
processedScenes = [mapName[len("map_"):-len(".png")] for mapName in os.listdir(STORAGE_DIR+"ilmMaps/") if mapName.startswith("map_")]
sceneNames = [sceneName[-4:] for sceneName in os.listdir(STORAGE_DIR) if sceneName.startswith("scene")]

for sceneName in sceneNames:
    if not(sceneName in processedScenes):
        print(sceneName)