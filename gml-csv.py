import networkx as nx
import csv

G = nx.read_gml("C:/Users/Bill/Desktop/499/dolphins.gml")

with open("dolphins.csv", "w", newline="") as output_file:
    csv_writer = csv.writer(output_file)
    csv_writer.writerow(["Node1", "Node2"])

    for edge in G.edges():
        csv_writer.writerow(edge)
