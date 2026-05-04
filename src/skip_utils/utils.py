def block_remove_llava(model, kill_list):
    
    kill_list.sort()

    while(len(kill_list)>0):
        
        del model.language_model.model.layers[kill_list[0]]
        del kill_list[0]
        for i in range(len(kill_list)):
            kill_list[i] -= 1

    for i in range(len(model.language_model.model.layers)):
        model.language_model.model.layers[i].self_attn.layer_idx = i

    return model

def block_remove_qwen2vl(model, kill_list):
    
    kill_list.sort()

    while(len(kill_list)>0):
        
        del model.model.language_model.layers[kill_list[0]]
        del kill_list[0]
        for i in range(len(kill_list)):
            kill_list[i] -= 1

    for i in range(len(model.model.language_model.layers)):
        model.model.language_model.layers[i].self_attn.layer_idx = i

    return model