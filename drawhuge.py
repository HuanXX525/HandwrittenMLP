from graphviz import Digraph

def trace(root):
    """保持原有迭代遍历逻辑不变"""
    nodes, edges = set(), set()
    stack = [root]
    nodes.add(root)
    while stack:
        v = stack.pop()
        for child in v._child:
            edges.add((child, v))
            if child not in nodes:
                nodes.add(child)
                stack.append(child)
    return nodes, edges

def draw_dot(root, format='pdf', rankdir='LR', filename='computation_graph'):
    """仅显示有运算符的节点的 op 字段，无运算符节点不画 op"""
    print(f"Drawing graph in format: {format}, rankdir: {rankdir}")
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    print(f"Nodes: {len(nodes)}, Edges: {len(edges)}")
    
    dot = Digraph(
        format=format,
        graph_attr={'rankdir': rankdir},
        node_attr={'shape': 'record'},
        filename=filename
    )
    
    # 动态生成节点标签：有运算符则显示 op，无则只显示 data 和 grad
    for n in nodes:
        if n._op:
            # 有运算符：op | data | grad
            dot.node(
                name=str(id(n)),
                label=f"{{ op: {n._op} | data: {n.data:.4f} | grad: {n.grad:.4f} }}",
                color="blue"
            )
        else:
            # 无运算符（原始输入）：data | grad
            dot.node(
                name=str(id(n)),
                label=f"{{ data: {n.data:.4f} | grad: {n.grad:.4f} }}",
                color="gray"  # 原始节点标灰，视觉区分
            )
    
    # 直接连接子节点和父节点，无冗余边
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)))
    
    dot.render(filename=filename, view=False)
    print(f"优化后PDF文件已保存为：{filename}.pdf")
    return dot