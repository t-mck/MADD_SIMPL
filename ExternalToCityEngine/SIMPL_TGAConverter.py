import os
from PIL import Image

class SIMPL_TGAConverter:
    def __init__(self):
        pass

    def convert_tga_to_jpg(self, tga_path):
        '''
        Using GIMP convert .tga files to .jpg
        :param tga_path:
        :return:
        '''
        try:
            img = Image.open(tga_path)
            new_path = tga_path.replace('.tga', '.png')
            new_path = new_path.replace('.TGA', '.png')
            new_path = new_path.replace('.Tga', '.png')
            img.save(new_path)
        except OSError as e:
            debug=1

    @staticmethod
    def get_all_files(directory_path):
        # List all files in the given directory
        filez = os.listdir(directory_path)
        files = []
        for f in filez:
            files.append(repr(f)[1:-1])
        return files

    def convert_all_tga_in_dir_to_jpg(self, top_dir_path):
        # get all asset dirs
        asset_dirs = self.get_all_files(top_dir_path)
        tga_paths = []
        #get all .tga paths
        for asset_dir in asset_dirs:
            tga_paths = self.get_single_asset_tga_paths_in_dir(asset_dir, top_dir_path, tga_paths)

        for tga_path in tga_paths:
            self.convert_tga_to_jpg(tga_path)

    def get_single_asset_tga_paths_in_dir(self, asset_dir, top_dir_path, tga_paths):
            ad = top_dir_path + asset_dir
            file_paths = self.get_all_files(ad)
            for fp in file_paths:
                if fp.endswith('.tga') or fp.endswith('.TGA') or fp.endswith('.Tga'):
                    tga_paths.append(ad + '/' + fp)

            return tga_paths

    def convert_single_asset_tag_in_dir_to_jpg(self, asset_dir, top_dir_path):
        tga_paths = []
        tga_paths = self.get_single_asset_tga_paths_in_dir(asset_dir, top_dir_path, tga_paths)
        for tga_path in tga_paths:
            self.convert_tga_to_jpg(tga_path)


def main():
    top_dir_path = 'C:/Users/tm-pc-win/Documents/AMLL/ONR_Code35/Working_CityEngine_Models/SK_East_APC_BRDM2/'
    stgac = SIMPL_TGAConverter()
    # stgac.convert_all_tga_in_dir_to_jpg(top_dir_path)
    stgac.convert_single_asset_tag_in_dir_to_jpg(asset_dir='SK_East_APC_BRDM2/', top_dir_path='C:/Users/tm-pc-win/Documents/AMLL/ONR_Code35/Working_CityEngine_Models/')

if __name__ == '__main__':
    main()
