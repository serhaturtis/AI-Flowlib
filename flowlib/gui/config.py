import os

BUILD_TYPE = "dev"
BUILD_VERSION = "0.0.1"
WHEELS_DIRECTORY = "wheels"
REQUIREMENTS_FILE = "requirements_with_versions.txt"

RESOURCES_FOLDER = os.path.join(os.path.dirname(__file__), "resources")

DEV_NAME = 'AI-Flowlib'
APP_NAME = 'Flowlib Configuration Manager'
APP_LOGO_SVG_PATH = os.path.join(RESOURCES_FOLDER, 'icon.svg')
APP_LOGO_ICO_PATH = os.path.join(RESOURCES_FOLDER, 'icon.ico')
APP_LOGO_PNG_PATH = os.path.join(RESOURCES_FOLDER, 'icon.png')

APP_MINIMIZE_ICON_PATH = os.path.join(RESOURCES_FOLDER, 'minimize.svg')
APP_RESTORE_ICON_PATH = os.path.join(RESOURCES_FOLDER, 'restore.svg')
APP_CLOSE_ICON_PATH = os.path.join(RESOURCES_FOLDER, 'close.svg')
PROJECT_NEW_ICON_PATH = os.path.join(RESOURCES_FOLDER, 'new.svg')
PROJECT_OPEN_ICON_PATH = os.path.join(RESOURCES_FOLDER, 'open.svg')
WORKSPACE_ICON_PATH = os.path.join(RESOURCES_FOLDER, 'open.svg')
PROJECT_REMOVE_ICON_PATH = os.path.join(RESOURCES_FOLDER, 'remove.svg')

NODE_WARNING_COLOR_A = (105, 0, 0)
NODE_WARNING_COLOR_B = (105, 105, 0)

IMGPROC_INPUT_NODE_COLOR = (0, 35, 105)
IMGPROC_AI_NODE_COLOR = (35, 105, 35)
IMGPROC_FILTER_NODE_COLOR = (35, 35, 35)
IMGPROC_OUTPUT_NODE_COLOR = (105, 35, 0)

SIGPROC_INPUT_NODE_COLOR = (0, 35, 105)
SIGPROC_AI_NODE_COLOR = (35, 105, 35)
SIGPROC_FILTER_NODE_COLOR = (35, 35, 35)
SIGPROC_OUTPUT_NODE_COLOR = (105, 35, 0)

LLM_TEMPLATE_NODE_COLOR = (35, 35, 35)
LLM_AGENT_NODE_COLOR = (35, 35, 35)
LLM_FLOW_NODE_COLOR = (35, 35, 35)
LLM_ACTION_NODE_COLOR = (35, 35, 35)

NOTICE_FILE = os.path.join(os.path.dirname(__file__), 'resources/notice.txt')
LCS_FILE = os.path.join(os.path.expanduser('~'), '.flowlib', 'license_accepted')
LOG_FILE = os.path.join(os.path.expanduser('~'), '.flowlib', 'logs', 'gui.log')

# Pretrained model paths:
PRETRAINED_MODELS_PATH = os.path.join(os.path.dirname(__file__), 'pretrained_models')
PRETRAINED_RESNET50_PATH = os.path.join(PRETRAINED_MODELS_PATH, 'image/resnet50/resnet50_imagenet_weights.pth')
PRETRAINED_IMAGENET_LABELS_PATH = os.path.join(PRETRAINED_MODELS_PATH, 'image/resnet50/imagenet_class_index.json')
PRETRAINED_BLIP_PROCESSOR_PATH = os.path.join(PRETRAINED_MODELS_PATH, 'image/blip/blip_processor')
PRETRAINED_BLIP_MODEL_PATH = os.path.join(PRETRAINED_MODELS_PATH, 'image/blip')