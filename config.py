
forward_models = [
    {
        "name": "Clean Coarse",
        "path_fwd": "forward_models/Clean_coarse-fwd.fif",
        "path_info": "forward_models/info.pkl",
        "distortion": "clean",
        "resolution": "coarse",
        "magnitude" : 0
    },
    {
        "name": "Clean Fine",
        "path_fwd": "forward_models/Clean_fine-fwd.fif",
        "path_info": "forward_models/info.pkl",
        "distortion": "clean",
        "resolution": "fine",
        "magnitude" : 0
    },
    {
        "name": "Altered Rot. Right 1°",
        "path_fwd": "forward_models/Altered_fine_rotation-1deg-right-fwd.fif",
        "path_info": "forward_models/info_rotation-1deg-right.pkl",
        "distortion": "rotation right",
        "resolution": "fine",
        "magnitude" : 1
    },
    {
        "name": "Altered Rot. Right 2°",
        "path_fwd": "forward_models/Altered_fine_rotation-2deg-right-fwd.fif",
        "path_info": "forward_models/info_rotation-2deg-right.pkl",
        "distortion": "rotation right",
        "resolution": "fine",
        "magnitude" : 2
    },
    {
        "name": "Altered Rot. Up 0.25°",
        "path_fwd": "forward_models/Altered_fine_rotation-025deg-up-fwd.fif",
        "path_info": "forward_models/info_rotation-025deg-up.pkl",
        "distortion": "rotation up",
        "resolution": "fine",
        "magnitude" : 0.25
    },
    {
        "name": "Altered Rot. Up 0.5°",
        "path_fwd": "forward_models/Altered_fine_rotation-05deg-up-fwd.fif",
        "path_info": "forward_models/info_rotation-05deg-up.pkl",
        "distortion": "rotation up",
        "resolution": "fine",
        "magnitude" : 0.5
    },
    {
        "name": "Altered Trans. Dorsal 1mm",
        "path_fwd": "forward_models/Altered_fine_translation-1mm-dorsal-fwd.fif",
        "path_info": "forward_models/info_translation-1mm-dorsal.pkl",
        "distortion": "translation dorsal",
        "resolution": "fine",
        "magnitude" : 1
    },
    {
        "name": "Altered Trans. Dorsal 2mm",
        "path_fwd": "forward_models/Altered_fine_translation-2mm-dorsal-fwd.fif",
        "path_info": "forward_models/info_translation-2mm-dorsal.pkl",
        "distortion": "translation dorsal",
        "resolution": "fine",
        "magnitude" : 2
    },
    {
        "name": "Altered Trans. Posterior 1mm",
        "path_fwd": "forward_models/Altered_fine_translation-1mm-posterior-fwd.fif",
        "path_info": "forward_models/info_translation-1mm-posterior.pkl",
        "distortion": "translation posterior",
        "resolution": "fine",
        "magnitude" : 1
    },
    {
        "name": "Altered Trans. Posterior 2mm",
        "path_fwd": "forward_models/Altered_fine_translation-2mm-posterior-fwd.fif",
        "path_info": "forward_models/info_translation-2mm-posterior.pkl",
        "distortion": "translation posterior",
        "resolution": "fine",
        "magnitude" : 2
    },
    {
        "name": "Altered Trans. Right 1mm",
        "path_fwd": "forward_models/Altered_fine_translation-1mm-right-fwd.fif",
        "path_info": "forward_models/info_translation-1mm-right.pkl",
        "distortion": "translation right",
        "resolution": "fine",
        "magnitude" : 1
    },
    {
        "name": "Altered Trans. Right 2mm",
        "path_fwd": "forward_models/Altered_fine_translation-2mm-right-fwd.fif",
        "path_info": "forward_models/info_translation-2mm-right.pkl",
        "distortion": "translation right",
        "resolution": "fine",
        "magnitude" : 2
    },
    
]

subject = "fsaverage"
subjects_dir = r"C:\Users\lukas\mne_data\MNE-sample-data\subjects"
