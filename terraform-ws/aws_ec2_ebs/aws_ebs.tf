resource "aws_ebs_volume" "volume1" {
    availability_zone= aws_instance.instance1.availability_zone
    size = var.ebs_size
    tags = {
        Name = "Python-Face-Recognition-Volume"
    }
}

resource "aws_volume_attachment" "volume1_attach" {
    device_name = "/dev/sdc"
    volume_id   = aws_ebs_volume.volume1.id
    instance_id = aws_instance.instance1.id
  
}