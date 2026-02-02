import sys
import os

# 添加项目源码路径
script_dir = os.path.dirname(os.path.abspath(__file__))
src_python_path = os.path.join(script_dir, '..', 'src', 'python')
sys.path.insert(0, src_python_path)

import prior
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from pipeline import pipeline, baseline

if __name__ == "__main__":

    print("Loading dataset...")
    # 使用默认 revision，与 ai2thor 5.0.0 兼容
    dataset = prior.load_dataset("procthor-10k")
    print("Dataset loaded!")
    
    print("Initializing AI2-THOR controller (this may take a while on first run)...")
    # 如果在无头服务器上运行，使用 platform=CloudRendering
    # 如果有显示器，可以注释掉 platform=CloudRendering
    controller = Controller(
        agentMode="default",
        renderInstanceSegmentation=True,
        renderDepthImage=True,
        width=800,
        height=600,
        visibilityDistance=2.5,
        # platform=CloudRendering,  # 无头服务器需要启用此选项
    )
    print("Controller initialized!")

    # 尝试找到一个有效的房屋
    house_idx = 10  # 初始房屋索引
    house_indices_to_try = [house_idx] + [i for i in range(0, 200, 10) if i != house_idx]
    house_instance = None
    reachable_positions = None

    for try_idx in house_indices_to_try:
        print(f"Trying house index {try_idx}...")
        house_instance = dataset["train"][try_idx]

        event = controller.reset(scene=house_instance)
        if not event.metadata.get('lastActionSuccess', True):
            print(f"  Reset failed: {event.metadata.get('errorMessage', 'Unknown error')}")
            continue

        event = controller.step(action="GetReachablePositions")
        reachable_positions = event.metadata["actionReturn"]

        if reachable_positions and len(reachable_positions) > 0:
            house_idx = try_idx
            print(f"Success! House {try_idx} has {len(reachable_positions)} reachable positions.")
            break
        else:
            print(f"  House {try_idx} has no reachable positions, skipping...")

    if reachable_positions is None or len(reachable_positions) == 0:
        print("ERROR: Could not find a valid house after trying multiple indices. Exiting.")
        controller.stop()
        sys.exit(1)

    print(f"Using house index: {house_idx}")
    print("Scene loaded successfully!")
    
    objects = ["AlarmClock", "Laptop", "CellPhone"]

    params = {}
    params["General"] = {}
    params["Agent"] = {}
    params["SceneGraph"] = {}

    ## General Parameters
    params["General"]["output_folder"] = "./output"  # 输出到当前目录下的 output 文件夹
    params["General"][
        "scene_graph_source"
    ] = 0  # 0: From Ground Truth Positions, 2: Visual Observations
    params["General"]["enable_llm_room_tracking"] = False
    # params["General"]["llm_model_name"] = "gpt-3.5-turbo"
    params["General"]["llm_model_name"] = "deepseek-chat"  # DeepSeek V3 模型
    params["General"][
        "openai_key"
    ] = "sk-c7d0d89a5f9f4d8fb4209109be981e1f"  # 填入你的 API Key
    params["General"][
        "openai_base_url"
    ] = "https://api.deepseek.com"  # DeepSeek API Base URL

    ## Low-Level Planner (Agent) Parameters
    params["Agent"]["grid_size"] = 0.25
    params["Agent"][
        "nav_policy_type"
    ] = 1  # 0: Jump from one point to another, 1: Oracle Planner, 2: PointNav Planner
    params["Agent"]["policy_model"] = "point_nav.pt"

    ## Scene Graph Parameters
    params["SceneGraph"]["min_pixels_per_object"] = 20
    params["SceneGraph"]["door_wall_matching_thresh_dist"] = 0.05

    ## Use either this:
    params["General"]["start_room_num"] = 2
    ## Or these:
    # params['General']['start_position'] = (1.0, 3.75)
    # params['General']['start_heading'] = 180

    ## Provide GT to compute SPL
    # params['General']['shortest_path_length'] = 10.0

    ## Run the pipeline
    pipeline(house_instance, controller, objects, params)
    print("Down!")
