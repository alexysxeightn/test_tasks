from collections import defaultdict, deque
import sys
from typing import List, Dict, Union


class Element:
    """
    Класс для представления элемента графа (узла или ребра)
    :param atr:          значение атрибута
    :param is_computed:  флаг вычисления
    :param dependencies: список зависимостей для вычисления
    :param func:         функция (min, *)
    """

    def __init__(self):
        self.atr: Union[float, str, None] = None
        self.is_computed: bool = False
        self.dependencies: List[int] = []
        self.func: Union[str, None] = None


class Graph:
    """
    Класс для представления графа
    :param nv:        количество вершин
    :param ne:        количество ребер
    :param vertices:  список вершин графа
    :param edges:     список ребер графа
    :param elements:  список вершин и ребер графа
    :param adj_edges: словарь узел -> смежные ребра
    :param edge_src:  словарь ребро -> исходный узел
    """

    def __init__(self, nv: int, ne: int):
        self.nv = nv
        self.ne = ne
        self.vertices: List[Element] = [Element() for _ in range(nv)]
        self.edges: List[Element] = [Element() for _ in range(ne)]
        self.elements: List[Element] = self.vertices + self.edges
        self.adj_edges: Dict[int, List[int]] = defaultdict(list)
        self.edge_src: Dict[int, int] = dict()

    def add_edge(self, edge_id: int, src: int, dst: int):
        """Добавляет ребро в граф"""
        src -= 1
        dst -= 1
        self.edge_src[edge_id] = src
        self.adj_edges[dst].append(edge_id)

    def get_adj_edges(self, vertex_id: int) -> List[int]:
        """Возвращает список смежных ребер для узла"""
        return self.adj_edges.get(vertex_id, [])

    def get_edge_src(self, edge_id: int) -> int:
        """Возвращает исходный узел для ребра"""
        return self.edge_src[edge_id]

    def compute_attribute(self, element_id: int) -> bool:
        """Вычисляет атрибут элемента, если возможно"""
        element = self.elements[element_id]

        if element.is_computed:
            return True

        # Проверить, все ли зависимости вычислены и не содержат None
        for dep in element.dependencies:
            if not self.elements[dep].is_computed or self.elements[dep].atr is None:
                return False  # Зависимость не готова

        if element.func == "min":
            adj_edges = self.get_adj_edges(element_id)
            values = [self.elements[self.nv + e].atr for e in adj_edges]
            if None in values:
                return False  # Не все ребра вычислены
            element.atr = min(values)
            element.is_computed = True
            return True
        elif element.func == "*":
            edge_id = element_id - self.nv
            src_vertex = self.get_edge_src(edge_id)
            src_val = self.elements[src_vertex].atr
            if src_val is None:
                return False  # Исходный узел не вычислен
            adj_edges = self.get_adj_edges(src_vertex)
            values = [self.elements[self.nv + e].atr for e in adj_edges]
            if None in values:
                return False  # Не все ребра вычислены
            product = src_val
            for val in values:
                product *= val
            element.atr = product
            element.is_computed = True
            return True
        elif element.dependencies:
            # Копирование значения из зависимости
            dep_value = self.elements[element.dependencies[0]].atr
            if dep_value is None:
                return False
            element.atr = dep_value
            element.is_computed = True
            return True

        return False

    def detect_cycle(self) -> bool:
        """Проверяет граф на наличие циклов"""
        visited = [False] * len(self.elements)
        rec_stack = [False] * len(self.elements)

        def dfs(node):
            if rec_stack[node]:
                return True
            if visited[node]:
                return False
            visited[node] = True
            rec_stack[node] = True
            for dep in self.elements[node].dependencies:
                if dfs(dep):
                    return True
            rec_stack[node] = False
            return False

        for i in range(len(self.elements)):
            if dfs(i):
                return True
        return False

    def compute_all_attributes(self) -> List[Element]:
        """Вычисляет все атрибуты графа и возвращает их"""
        if self.detect_cycle():
            raise ValueError("Ошибка: Циклическая зависимость в правилах.")

        stack = deque(range(self.nv + self.ne))
        attempts = 0
        max_attempts = 1000  # Ограничение на количество попыток

        while stack and attempts < max_attempts:
            element_id = stack.popleft()
            if self.compute_attribute(element_id):
                # Добавить зависящие элементы в стек
                for i, el in enumerate(self.elements):
                    if not el.is_computed and element_id in el.dependencies:
                        if i not in stack:
                            stack.append(i)
            else:
                # Вернуть элемент в конец стека, если не готов
                stack.append(element_id)
            attempts += 1

        if attempts >= max_attempts:
            raise ValueError("Ошибка: Невозможно вычислить атрибуты всех элементов.")

        return self.elements


def parse_input(file_path: str) -> Graph:
    """Чтение входного файла и создание графа"""
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    nv, ne = map(int, lines[0].split())
    graph = Graph(nv, ne)
    idx = 1

    for i in range(ne):
        src, dst = map(int, lines[idx].split())
        graph.add_edge(i, src, dst)
        idx += 1

    for i in range(nv + ne):
        rule = lines[idx]
        element = graph.vertices[i] if i < nv else graph.edges[i - nv]

        if rule in ("min", "*"):
            # Вычислимая функция
            element.func = rule
        elif rule.startswith("e"):
            # Зависимость от ребра: e <номер>
            ref_id = int(rule.split()[1]) - 1
            element.dependencies.append(nv + ref_id)  # Ребра идут после узлов
        elif rule.startswith("v"):
            # Зависимость от узла: v <номер>
            ref_id = int(rule.split()[1]) - 1
            element.dependencies.append(ref_id)
        else:
            # Числовое значение
            element.atr = float(rule)
            element.is_computed = True
        idx += 1

    return graph


def write_result(output_file: str, elements: List[Element]):
    """Записывает полученные атрибуты всех элементов в файл"""
    with open(output_file, "w") as f:
        f.write("\n".join([str(el.atr) for el in elements]))


def main(input_file: str, output_file: str):
    try:
        graph = parse_input(input_file)
        elements = graph.compute_all_attributes()
        write_result(output_file, elements)
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])