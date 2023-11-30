import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import urllib.parse
from community import community_louvain

# Streamlit 사이드바에 메뉴 추가
with st.sidebar:
    page = option_menu("App Gallery", ["MapY", "서비스 소개", "네트워크 시각화"],
                         icons=['house', 'card-list', 'bezier2'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "4!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#fafafa"},
        "nav-link-selected": {"background-color": "#08c7b4"},
    }
    )

if page == "MapY":
    # title과 subheader를 중앙 정렬하는 CSS 스타일을 적용한 코드
    #st.title("MapY")
    #st.subheader("유튜브 토픽 매핑 서비스")
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>MapY</h1>
            <h3>유튜브 토픽 매핑 서비스</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 이미지 파일 경로
    image_path = "image1.png"  # 이미지 파일의 경로를 지정해야 합니다.
    # 이미지 표시
    st.image(image_path, use_column_width=True)

    st.markdown(
        """
        <div style='text-align: center; color: grey;'>
            MapY는 네트워크 분석을 통해 유튜브 댓글 데이터를 시각화하는 서비스입니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

if page == "서비스 소개":
    st.title("서비스 소개")
    st.subheader("반복적인 유튜브 알고리즘, 지겹지 않으셨나요?")
    st.write("MapY는 네트워크 분석을 통해 유튜브 댓글 데이터를 시각화하는 서비스입니다.")
    st.write("사용자가 원하는 키워드로 유튜브에서 동영상을 검색하고, 해당 검색어와 관련된 네트워크를 구축해주는 솔루션입니다.")
    st.write("사용자가 입력한 검색어를 기반으로 유튜브에서 관련 동영상을 검색하여, 이를 시각적으로 연결된 네트워크 그래프로 제시합니다. ")

    image_path3 = "image2.png"  # 이미지 파일의 경로를 지정해야 합니다.
    # 이미지 표시
    st.image(image_path3, use_column_width=True)
    st.write(" ")
    st.write(" ")

    st.subheader("유튜브의 최근 트렌드를 알아보고 싶지 않으신가요?")
    st.write("MapY는 최근 한 달 동안 업로드된 영상들을 기반으로, 검색어와 관련된 다양한 주제, 콘텐츠, 및 연관 키워드들을 시각적으로 파악할 수 있습니다.")
    st.write("개인이 아닌 유튜브 사용자 전체의 의견과 관심사를 이해하고, 그들이 어떤 주제에 관심을 가지고 있는지 효과적으로 보여줍니다.")
    st.write("MapY를 통해 새로운 트렌드나 주목할만한 주제를 발견하는데 도움이 되며, 이로부터 유용한 인사이트를 제공하여 컨텐츠 제작 및 마케팅 전략에 활용될 수 있을 것입니다.")
    image_path2 = "image3.jpg"
    # 이미지 표시
    st.image(image_path2, use_column_width=True)



if page == "네트워크 시각화":

    # 데이터 경로 설정
    data_paths = {
        "Economy": 'e_t_50.csv',
        "Fashion": 'f_t_50.csv',
        "Science": 's_t_50.csv'
    }

    # 기본 데이터 선택
    default_data = "Economy"

    # 데이터 불러오기 함수
    @st.cache_data
    def load_data(path):
        return pd.read_csv(path, index_col=0)

    # 네트워크 그래프 생성 함수
    @st.cache_data
    def create_graph(df):
        G = nx.Graph()

        # 노드 추가
        for key in df.index.tolist():
            G.add_node(key)

        # 엣지 임계값 생성
        flat_data = df.values.flatten()
        q1 = np.percentile(flat_data, 50)

        # 엣지 추가
        for i in df.columns:
            for j in df.columns:
                if i != j:
                    if df[i][j] > q1:
                        G.add_edge(i, j, weight=df[i][j])
                    else:
                        pass
                else:
                    pass

        return G

    # 그래프 시각화 함수
    def visualize_graph(G, pos, node_sizes, node_colors, edge_trace, node_trace):
        # 노드 크기에 따른 텍스트 크기 계산
        text_sizes = [size / 4 for size in node_sizes]

        # 노드 텍스트 트레이스 추가
        text_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            text=[str(node) for node in G.nodes()],
            mode='text',
            hoverinfo='none',
            textposition='middle center',
            textfont=dict(
                size=text_sizes,  # 동적 텍스트 크기
                color='black',
                family='Roboto Bold'
            )
        )

        # Plotly 그래프 객체 생성
        fig = go.Figure(data=[edge_trace, node_trace, text_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=0),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                            )
                        )
        
        # 중복 색상을 제거하고 순서를 유지하기 위해 OrderedDict 사용
        from collections import OrderedDict

        # color_lst에 있는 색상 중에서 고유한 색상만 추출
        unique_colors = list(OrderedDict.fromkeys(color_lst))

        # 커뮤니티 이름과 색상을 매핑
        community_names = {
        'red': 'Group 1',
        'green': 'Group 2',
        'blue': 'Group 3',
        'yellow': 'Group 4',
        '#3F7CAC': 'Group 5',
        '#B6A2DE': 'Group 6',
        '#FE5F55': 'Group 7'
        }

        # 커뮤니티 이름과 색상을 매핑하기 위한 딕셔너리 생성
        color_to_community = {color: community_names[color] for color in unique_colors if color in community_names}

        
        # 범례 추가
        for color, community_name in color_to_community.items():
            # 주석을 통해 범례를 그래프에 추가
            fig.add_annotation(
                xref='paper', yref='paper',
                x=0.05, y=0.95 - unique_colors.index(color) * 0.05,  # 위치 조정
                text=community_name,  # 커뮤니티 이름
                showarrow=False,
                font=dict(
                    family='Arial',
                    size=12,
                    color=color  # 커뮤니티 색상
                ),
                align='left',
                xanchor='left',
                yanchor='top',
                bgcolor='white',  # 배경색
                bordercolor='black',  # 테두리 색
                borderwidth=1
            )


        # 마우스 호버 시에 노드 강조 효과
        fig.update_traces(marker=dict(line=dict(color='black', width=1)))
        return fig

    # URL에 사용할 수 있도록 노드 이름을 변환하는 함수
    def convert_to_url_format(node_name):
        return urllib.parse.quote_plus(node_name)

    # 메인 코드
    st.title("MapY")
    st.write("이 앱은 네트워크 분석을 통해 유튜브 댓글 데이터를 시각화하는 서비스입니다.")
    st.write("")

    if 'current_category' not in st.session_state:
        st.session_state.current_category = "Economy"

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_data_Economy = st.button("Economy", key="Economy")
        if selected_data_Economy:
            st.session_state.current_category = "Economy"

    with col2:
        selected_data_Fashion = st.button("Fashion", key="Fashion")
        if selected_data_Fashion:
            st.session_state.current_category = "Fashion"

    with col3:
        selected_data_Science = st.button("Science", key="Science")
        if selected_data_Science:
            st.session_state.current_category = "Science"

    st.write("")

    df_selected = load_data(data_paths[st.session_state.current_category])
    G_selected = create_graph(df_selected)

    num_nodes_to_display = st.slider("표현할 노드 수 선택", min_value=1, max_value=len(df_selected), value=10)

    pos = nx.spring_layout(G_selected)
    selected_nodes = sorted(G_selected.nodes, key=lambda x: G_selected.degree[x], reverse=True)[:num_nodes_to_display]
    G_filtered = G_selected.subgraph(selected_nodes)

    node_degrees = nx.degree_centrality(G_selected)
    node_list=[node_degrees[node] for node in G_filtered]
    min_node = min(node_list)
    max_node = max(node_list)
    node_sizes=[]
    for i in node_list:
        j=(((i-min_node)/(max_node-min_node))*3+1)*30
        node_sizes.append(j)

    # G_filtered에 대한 커뮤니티 파티션 정보 생성
    partition = community_louvain.best_partition(G_filtered)

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in G_filtered.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    node_colors = ['red', 'green', 'blue', 'yellow', '#3F7CAC', '#B6A2DE', '#FE5F55']
    color_lst = []
    for value in partition.values():
        color_lst.append(node_colors[value])

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            symbol='hexagon',
            size=node_sizes,
            color=color_lst,  # partition 정보를 이용해 노드 색상 지정
            showscale=False,
            colorscale='YlGnBu',
            colorbar=dict(
                thickness=10,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )

    for node in G_filtered.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([node])

    # 그래프 표시
    st.plotly_chart(visualize_graph(G_filtered, pos, node_sizes, color_lst, edge_trace, node_trace))

    # 선택된 노드와 연결된 노드들을 선택하는 상자 추가
    connected_nodes = []
    selected_node = st.selectbox("선택된 키워드", list(G_filtered.nodes()))
    subgraph_nodes = list(G_filtered.neighbors(selected_node)) + [selected_node]
    subgraph = G_filtered.subgraph(subgraph_nodes)

    top_sub_pos = pos.copy()
    top_connected_nodes = sorted(subgraph.neighbors(selected_node), key=lambda x: subgraph.degree[x], reverse=True)[:9]
    top_connected_subgraph = subgraph.subgraph([selected_node] + top_connected_nodes)

    node_degrees_1 = nx.degree_centrality(G_selected)
    node_list=[node_degrees_1[node] for node in top_connected_subgraph.nodes()]
    min_node = min(node_list)
    max_node = max(node_list)
    node_sizes=[]
    for i in node_list:
        j=(((i-min_node)/(max_node-min_node))*3+1)*30
        node_sizes.append(j)

    top_sub_edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in top_connected_subgraph.edges():
        x0, y0 = top_sub_pos[edge[0]]
        x1, y1 = top_sub_pos[edge[1]]
        top_sub_edge_trace['x'] += tuple([x0, x1, None])
        top_sub_edge_trace['y'] += tuple([y0, y1, None])

    top_sub_node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            symbol='hexagon',
            size=node_sizes,
            color=[],
            colorscale='YlGnBu',
            line=dict(color='black', width=2)))

    for node in top_connected_subgraph.nodes():
        x, y = top_sub_pos[node]
        top_sub_node_trace['x'] += tuple([x])
        top_sub_node_trace['y'] += tuple([y])
        top_sub_node_trace['text'] += tuple([node])

        # 커뮤니티 정보를 이용하여 색상 지정
        community_color = partition[node]
        if node == selected_node:
            top_sub_node_trace['marker']['color'] += tuple(['red'])
        else:
            top_sub_node_trace['marker']['color'] += tuple([community_color])

    # 상위 9개 노드로 구성된 부분 그래프 그리기
    st.plotly_chart(visualize_graph(top_connected_subgraph, top_sub_pos, node_sizes, list(partition.values()), top_sub_edge_trace, top_sub_node_trace))

    # 선택된 노드와 연결된 노드들을 선택하는 상자 추가
    connected_nodes = []
    selected_connected_nodes = []
    if selected_node:
        connected_nodes = list(G_filtered.neighbors(selected_node))
        key_connected_nodes = f"selected_connected_nodes_{selected_node}"
        selected_connected_nodes = st.multiselect("연결된 키워드 선택", connected_nodes, key=key_connected_nodes)


    # 새로운 그래프를 만들어 선택된 노드와 연결된 노드만 포함하도록 함
    subgraph_nodes = [selected_node] + selected_connected_nodes
    connected_subgraph = G_filtered.subgraph(subgraph_nodes)

    # 새로운 그래프의 커뮤니티 파티션 정보 생성
    connected_partition = community_louvain.best_partition(connected_subgraph)

    # 그래프 시각화
    edge_trace_connected = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in connected_subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace_connected['x'] += tuple([x0, x1, None])
        edge_trace_connected['y'] += tuple([y0, y1, None])

    node_trace_connected = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            symbol='pentagon',
            size=node_sizes,
            color=list(connected_partition.values()),  # 새로운 파티션 정보를 이용해 노드 색상 지정
            showscale=True,
            colorscale='YlGnBu',
            colorbar=dict(
                thickness=10,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )

    for node in connected_subgraph.nodes():
        x, y = pos[node]
        node_trace_connected['x'] += tuple([x])
        node_trace_connected['y'] += tuple([y])
        node_trace_connected['text'] += tuple([node])

    search_query = [selected_node] if selected_node else []
    search_query.extend(selected_connected_nodes)


    if search_query:
        category_prefix = {
            "Economy": "경제",
            "Science": "과학",
            "Fashion": "패션"
        }

        category = st.session_state.current_category
        category_query = category_prefix.get(category, '')
        full_query = [category_query] + search_query  # 카테고리 접두어를 검색어 리스트 앞에 추가
        query = 'AND'.join(full_query)  # 모든 검색어를 '.'으로 연결
        query = convert_to_url_format(query)
        youtube_search_url = f"https://www.youtube.com/results?search_query={query}"

        st.markdown(f'<a href="{youtube_search_url}" target="_blank"><button style="color: black; background-color: #FFFFFF; border: 2px solid black; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer;">유튜브 검색</button></a>', unsafe_allow_html=True)
