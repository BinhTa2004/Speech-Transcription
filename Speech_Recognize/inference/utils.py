    # Load weight
    # Load raw state_dict
    # Hàm để xóa 'module.' bởi vì model được train trên nhiều GPU
def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_key = k[7:]  # bỏ 'module.' đi
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict