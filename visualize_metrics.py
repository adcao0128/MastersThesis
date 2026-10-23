checkpoint = torch.load(os.path.join(save_path, 'gnn_model_valid_best.ckpt'))
model.load_state_dict(checkpoint['state_dict'])
test(model, graph, graph.test_mask, device,batch, p_x_all, p_edge_all)