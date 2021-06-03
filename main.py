import Utils

version = "0.0.2"


def train_and_augmente(GAN, dataset,
                       augmentation_percentage_list, if_keep_raw, advanced):
    if GAN == 'raw':
        return dataset
    elif GAN == 'gan':
        import gans.gan as gan
        return gan.trainGAN(dataset, augmentation_percentage_list,
                            if_keep_raw, advanced)
    elif GAN == 'dcgan':
        import gans.dcgan as dcgan
        return dcgan.trainGAN(dataset,
                              augmentation_percentage_list, if_keep_raw)
    elif GAN == 'wgan':
        import gans.wgan as wgan
        return wgan.trainGAN(dataset, augmentation_percentage_list,
                             if_keep_raw, advanced)
    elif GAN == 'tsagan':
        import gans.tsa_gan as tsagan
        return tsagan.trainGAN(dataset, augmentation_percentage_list,
                               if_keep_raw, advanced)
    else:
        print("GAN类别错误")
        exit(1)


def adjust_augmentation(method, tags):
    if method == 'auto':
        return Utils.adjust_augmentation_auto(tags)
    else:
        print("增强参数错误")
        exit(1)


def main():
    """
    读取json
    """
    config = Utils.readjson("GANs/setting.json")
    """
    版本检查
    """
    print("本地版本号:" + version)
    print("json版本号" + config["general"]["version"])
    if(1):
        print("版本兼容")
    else:
        print("版本不兼容")
    """
    读取数据
    """
    dataset = Utils.read_dataset(
        config["data"]["dataset_dir"], config["data"]["dataset_name"])
    dataset_name = config["data"]["dataset_name"]
    """
    配置通用参数
    """
    augmentation_percentage_method = config["general"]["augmentation_method"]
    augmentation_percentage_list = adjust_augmentation(
        augmentation_percentage_method, list(dataset[dataset_name][1]))
    if_keep_raw = config["general"]["keep_raw"]
    output_dir = config["data"]["output_dir"]
    """
    配置模型超参数
    """
    advanced = config["advanced"]
    """
    数据增强
    """
    new_dataset, training_information = train_and_augmente(
        config["general"]["model"],
        dataset, augmentation_percentage_list,
        if_keep_raw,
        advanced)
    """
    保存结果
    """
    Utils.savetofile(new_dataset, dataset_name, output_dir)


if __name__ == '__main__':
    main()
