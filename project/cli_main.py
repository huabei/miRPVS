'''
pytorch lightning cli main file
'''

from lightning.pytorch.cli import LightningCLI, ArgsType
from models import *
from models import PLBaseModel
from datamodule import DInterface
from time import strftime, localtime
# 获取当前时间
CURRENT_TIME = strftime("%Y-%m-%d-%H-%M-%S", localtime())

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument('--project', default='test', type=str)
        parser.add_argument('--group', default=None, type=str)
        parser.add_argument('--wandb', default=False, type=bool)
        parser.add_argument('--comment', default=None, type=str)
        # parser.link_arguments('trainer.default_root_dir', 'trainer.logger[0].init_args.save_dir')
        # parser.link_arguments('trainer.default_root_dir', 'trainer.logger[1].init_args.dir')       
        parser.link_arguments(['model.class_path', 'data.dataset'], 'trainer.default_root_dir', compute_fn=lambda x1, x2: 'log/' + x1.split('.')[-1] + '/' + x2)
        # parser.link_arguments('comment', 'trainer.logger[1].init_args.notes')
        # parser.link_arguments('project', 'trainer.logger[1].init_args.project')
        # parser.link_arguments('group', 'trainer.logger[1].init_args.group')
        
    def before_instantiate_classes(self) -> None:
        
        if self.config['subcommand'] == 'fit':
            if self.config.fit.wandb:
                import wandb
                wandb.login(key='local-8fe6e6b5840c4c05aaaf6aac5ca8c1fb58abbd1f', host='http://localhost:8080')
                self.config.fit.trainer.logger[1].init_args.dir = self.config.fit.trainer.default_root_dir
                self.config.fit.trainer.logger[1].init_args.project = self.config.fit.project
                self.config.fit.trainer.logger[1].init_args.notes = self.config.fit.comment
                self.config.fit.trainer.logger[1].init_args.group = self.config.fit.group
            else:
                # remove wandblogger
                self.config.fit.trainer.logger.pop()
            # overwrite logger init_args
            self.config.fit.trainer.logger[0].init_args.save_dir = self.config.fit.trainer.default_root_dir
            # checkpoints name
            if self.config.fit.trainer.enable_checkpointing:
                self.config.fit.trainer.callbacks[0].init_args.filename = CURRENT_TIME + '-best-{epoch}-{val_loss:.2f}'
            else:
                self.config.fit.trainer.callbacks.pop(0)
        # raise Exception('test')
    
    def before_fit(self) -> None:
        print(self.config)
        # raise Exception('test')


def cli_main(args: ArgsType = None):
    cli = MyLightningCLI(datamodule_class=DInterface,
                       subclass_mode_model=PLBaseModel,
                       seed_everything_default=1234,
                       auto_configure_optimizers=False,
                       args=args)
    # note: don't call fit!!



if __name__ == "__main__":
    # cli_main(['fit', '-c', 'project/config/pl_cli_config.yaml'])
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block

