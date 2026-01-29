from airspace import Airspace
import vp_design_GNN_RL_sb3_v2 as v

airspace = Airspace(100, 'Austin, Texas, USA', vertiport_tag_list=[('building','commercial')])

airspace.make_regions_dict_vp_des_test_mode(map_centeroid_to_region_center=2*(32_000_000**0.5), region_center_to_vp=32_000_000**0.5)

vp_graph_builder = v.VertiportGraphBuilder(airspace, connectivity_type='full')
vp_env = v.VertiportDesignEnv(2, 5, edge_feat_dim=2)

print(f'vertiport to idx: {vp_graph_builder.vertiport_to_idx}')
print(f'id to vertiport: {vp_graph_builder.idx_to_vertiport}')
print('region_mask: ')
print(vp_graph_builder.region_mask)


print('Observation space: ')
print('obs space - shape:')
sample_obs = vp_env.observation_space.sample()
sample_node_feat = sample_obs['node_feat']
sample_edge_attr = sample_obs['edge_attr']
sample_edge_index = sample_obs['edge_index']
sample_action_index = sample_obs['selected_actions']
print(f'node feat shape: {sample_node_feat.shape}')
print(f'edge_attr shape: {sample_edge_attr.shape}')
print(f'edge index shape: {sample_edge_index.shape}')
print(f'action index shape: {sample_action_index.shape}')


action_sample = vp_env.action_space.sample()
print(f'sample action: {action_sample}')



print('reset vp design env ')
reset_obs, reset_info = vp_env.reset()

#print(f'reset obs: {reset_obs["node_feat"]}')
print(f'reset obs - edge_attr: \n{reset_obs["edge_attr"]}')
print(f'edge_attr shape: \n{reset_obs["edge_attr"].shape}')

print()
print(f'reset obs - selected_actions: \n{reset_obs["selected_actions"]}')
print(f'reset obs - selected_actions shape: \n{reset_obs["selected_actions"].shape}')
print()

#print(f'reset info: {reset_info}')
print()
print(f'current_selected_vertiports: {vp_env.current_selected_vertiports}')
print()
print(f'current_metrics: \n{vp_env.current_metrics}')


print()
print()
print('Lets evaluate vp design system with sample action')
print()
print()

some_action = vp_env.action_space.sample()
print(f'some action: {some_action}')
obs, reward, terminated, trucated, info = vp_env.step(some_action)

print()
print(f'step obs - selected_actions: \n{obs["selected_actions"]}')
print(f'step obs - selected_actions shape: \n{obs["selected_actions"].shape}')
print()

print()
print(f'previous_selected_vertiports: {vp_env.previous_selected_vertiports}')
print()
print(f'current_selected_vertiports: {vp_env.current_selected_vertiports}')
print()
print(f'Observation node_feat: {obs["node_feat"]}')




