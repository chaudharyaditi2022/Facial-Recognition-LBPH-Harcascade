resource "aws_instance" "instance1" {
    ami = "ami-0ad704c126371a549"
    instance_type = var.instance_type
    tags = {
        Name = "Python-Face-Recognition-OS"
    }
}