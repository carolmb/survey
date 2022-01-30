import json
import requests


def download_net(net, subnet):
    pass
    # path = 'https://networks.skewed.de'
    # request_url = '%s/net/%s/files/%s.csv.zip' % (path, net, subnet)
    # response = requests.get(request_url)
    # if response.status_code == 200:
    #     content = response.content
    #
    #     file = open('data/%s_%s.csv.zip' % (net, subnet), 'wb')
    #     file.write(content)
    #     file.close()


if __name__ == '__main__':

    response = requests.get('https://networks.skewed.de/api/nets?tags=Weighted')
    if response.status_code == 200:
        nets = json.loads(response.content.decode('utf-8'))
        for net in nets:
            infos = requests.get('https://networks.skewed.de/api/net/%s' % net)
            if infos.status_code != 200:
                continue

            net_attributes = json.loads(infos.content.decode('utf-8'))

            if len(net_attributes['nets']) == 1:
                if net_attributes['analyses']['is_directed'] > 0:
                    print(net)

                    download_net(net, net)
            else:
                # print(net, len(net_attributes['analyses']))
                for subnet_name, subnet_info in net_attributes['analyses'].items():
                    if subnet_info['is_directed']: #and len(subnet_info['edge_properties']) > 0:
                        print(net)
                        download_net(net, subnet_name)
