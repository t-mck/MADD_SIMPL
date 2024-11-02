import bpy
import math
import os
from bpy.app.handlers import persistent

class SIMPL_Blender_Object_Rotater:
    def __init__(self):
        pass

    @staticmethod
    @persistent
    def load_handler(dummy):
        print("Load Handler:", bpy.data.filepath)

    def import_blend(self, target_file):
        bpy.app.handlers.load_post.append(self.load_handler) # this keeps the scene from changing when the file is loaded
        bpy.ops.wm.open_mainfile(filepath=target_file)

    def import_obj(self,
                   target_file):
        bpy.ops.wm.obj_import(filepath=target_file)

    def rotate_object(self,
                      offset = 90,
                      mesh_to_rotate='SkeletalMeshComponent0',
                      axis='z'):
        bpy.context.view_layer.objects.active = bpy.data.objects[mesh_to_rotate]
        obj = bpy.context.object
        if axis == 'z':
            obj.rotation_euler.z += math.radians(offset)
        elif axis == 'y':
            obj.rotation_euler.y += math.radians(offset)
        elif axis == 'x':
            obj.rotation_euler.x += math.radians(offset)
        else:
            raise ValueError('Axis must be z, y, or x')

    def export_obj(self,
                   directory,
                   fn='partially_rotated_object.obj'):
        # blend_file_path = bpy.data.filepath
        # directory = os.path.dirname(blend_file_path)
        target_file = os.path.join(directory, fn)
        bpy.ops.wm.obj_export(filepath=target_file)

    def make_rotated_obj_files(self,
                               start=0,
                               stop=91,
                               step=1,
                               axis='z',
                               dir_path = 'blend_and_obj_file_directory',
                               part_to_rotate_blend_file = 'part_to_rotate.blend',
                               main_body_obj_file = 'main_body.obj',
                               name_of_mesh_to_rotate='SkeletalMeshComponent0', # this can be found in the blender scene collection window, which is usually in the upper left hand corner of the screen (a typical default name is SkeletalMeshComponent0)
                               take_screen_shot=False
                               ):
        self.import_blend(target_file=dir_path + part_to_rotate_blend_file)
        self.import_obj(target_file=dir_path +  main_body_obj_file)
        for offset in range(start, stop, step):
            self.rotate_object(offset=offset, mesh_to_rotate=name_of_mesh_to_rotate, axis=axis)
            fn = f'part_{axis}_rotated_{offset}_degrees_plus_main_body.obj'
            self.export_obj(directory=dir_path,fn=fn)
            if take_screen_shot:
                self.take_screenshot_of_model(dir_path=dir_path, axis=axis, offset=offset)
            self.rotate_object(offset=offset*-1, mesh_to_rotate=name_of_mesh_to_rotate, axis=axis) # Reset the model

    def take_screenshot_of_model(self,
                                 dir_path,
                                 axis,
                                 offset):

        # Get the active camera
        camera = bpy.context.scene.camera

        # Render the scene
        bpy.ops.render.render()

        fn = f'part_{axis}_rotated_{offset}_degrees_plus_main_body.png'

        # Save the rendered image
        bpy.data.images['Render Result'].save_render(filepath=dir_path + fn)

def main():
    sbor = SIMPL_Blender_Object_Rotater()
    sbor.make_rotated_obj_files(start=0,
                                stop=360,
                                step=1,
                                axis='z',
                                dir_path="/home/taylor/Duke/AMLL/Synthetic_Data_Generation/pythonProject1/blender/SK_East_SPG_2S3_Akatsia/",
                                part_to_rotate_blend_file='SK_East_SPG_2S3_Akatsia_Turret_Plus.blend', #this part will be rotated around its origin/pivot point
                                main_body_obj_file='SK_East_SPG_2S3_Akatsia_MB.obj',                   #this part will stay in place
                                take_screen_shot=False
                                )


if __name__ == '__main__':
    main()
