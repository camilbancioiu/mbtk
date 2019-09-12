from pathlib import Path
import shutil

from mbff.utilities.Exceptions import LockablePathException


class LockablePath:

    def __init__(self, repository, folder):
        self.path = Path(repository, folder)
        self.default_lock_type = 'folder'


    def get_lock(self, lock_type=''):
        if lock_type == '': lock_type = self.default_lock_type
        return self.path / ('locked_{}'.format(lock_type))


    def folder_is_locked(self, lock_type=''):
        if lock_type == '': lock_type = self.default_lock_type
        return self.get_lock(lock_type).exists()


    def folder_exists(self):
        return self.path.exists()


    def subfolder_exists(self, subfolder):
        return (self.path / subfolder).exists()


    def lock_folder(self, lock_type=''):
        if lock_type == '': lock_type = self.default_lock_type
        if self.folder_is_locked(lock_type):
            pass
        else:
            self.get_lock(lock_type).write_text('locked')


    def delete_folder(self):
        if not self.folder_exists():
            raise LockablePathException(self, self.path, 'Folder {} does not exist, cannot delete it.'.format(self.name))
        if self.folder_is_locked():
            raise LockablePathException(self, self.path, 'Folder {} is locked, cannot delete it.'.format(self.name))
            return
        shutil.rmtree(self.path)


    def delete_subfolder(self, subfolder):
        if self.folder_is_locked():
            raise LockablePathException(self, self.path, 'Folder is locked, cannot delete any subfolder.')
        if not self.subfolder_exists(subfolder):
            raise LockablePathException(self, self.path, 'Fubfolder {} does not exist.'.format(subfolder))
        shutil.rmtree(self.path / subfolder)


    def unlock_folder(self, lock_type=''):
        if lock_type == '': lock_type = self.default_lock_type
        if not self.folder_is_locked(lock_type):
            pass
        else:
            self.get_lock(lock_type).unlink()


    def ensure_folder(self):
        self.path.mkdir(parents=True, exist_ok=True)


    def ensure_subfolder(self, subfolder_name):
        subfolder = (self.path / subfolder_name)
        subfolder.mkdir(parents=True, exist_ok=True)
        return subfolder


    def subfolder(self, subfolder_name):
        return self.ensure_subfolder(subfolder_name)
