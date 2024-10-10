from enum import Enum 

def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif'])

class GenerationMode(Enum):
    VQA = "vqa"
    VQA_NR = "vqa_nr"
    TQA = "tqa"
    DESCRIPT = "descript"
    GENERIC = "generic"
    VQA_TASK_DESC = "vqa_task_desc"
    DESCRIPT_TASK_DESC = "descript_task_desc"
    
def should_include_icl(mode: GenerationMode):
    if mode in [GenerationMode.VQA, GenerationMode.VQA_NR, GenerationMode.TQA, GenerationMode.DESCRIPT]:
        return True
    else:
        return False