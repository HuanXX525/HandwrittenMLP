from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._child:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='pdf', rankdir='LR', filename='computation_graph'):
    """
    输出计算图为 PDF 文件
    :param root: 计算图根节点（如 loss Value 对象）
    :param format: 输出格式，设为 'pdf' 即可
    :param rankdir: 布局方向（LR=左右，TB=上下）
    :param filename: 输出文件名（无需加 .pdf 后缀）
    :return: Digraph 对象，调用 render() 生成文件
    """
    print(f"Drawing graph in format: {format}, rankdir: {rankdir}")
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    print(f"Nodes: {len(nodes)}, Edges: {len(edges)}")
    
    # 关键修改：format 设为 'pdf'
    dot = Digraph(
        format=format,
        graph_attr={'rankdir': rankdir},
        node_attr={'shape': 'record'},  # 固定节点形状为 record
        filename=filename  # 指定输出文件名（后续无需重复写）
    )
    
    for n in nodes:
        # 节点标签：显示 data 和 grad（保留4位小数）
        dot.node(
            name=str(id(n)),
            label=f"{{ data: {n.data:.4f} | grad: {n.grad:.4f} }}"
        )
    
    for n1, n2 in edges:
        # 绘制节点间的边（子节点 → 父节点，符合计算图方向）
        dot.edge(str(id(n1)), str(id(n2)))
    
    # 生成 PDF 文件（默认保存在当前目录）
    dot.render(filename=filename, view=False)  # view=True 会自动打开 PDF
    print(f"PDF 文件已保存为：{filename}.pdf")
    return dot