"""Provision and maintain local research portal accounts from a terminal."""

import argparse
from getpass import GetPassWarning, getpass
import hashlib
import hmac
import re
import secrets
import sys
import warnings

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from .database import DatabaseSettings, PortalDatabase
from .models import PortalSession, PortalUser


ROLES = ('research_admin', 'research_operator', 'clinician_reviewer', 'research_viewer')
MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 512
_USERNAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}')
_FORMAT = 'scrypt-32768-8-3'


def _derive(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode('utf-8'), salt=salt, n=32768, r=8, p=3, dklen=32,
                          maxmem=64 * 1024**2)


def hash_password(password: str) -> str:
    if not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH:
        raise ValueError(f'Enter a password between {MIN_PASSWORD_LENGTH} and {MAX_PASSWORD_LENGTH} characters.')
    salt = secrets.token_bytes(16)
    try:
        return f'{_FORMAT}${salt.hex()}${_derive(password, salt).hex()}'
    except UnicodeEncodeError:
        raise ValueError('Enter a password with valid Unicode characters.') from None


def verify_password(password: str, stored_hash: str | None) -> bool:
    if stored_hash is None:
        # Pay the same password derivation cost for accounts with no credentials.
        _derive(password, bytes(16))
        return False
    try:
        version, salt, digest = stored_hash.split('$')
        if version != _FORMAT:
            return False
        return hmac.compare_digest(_derive(password, bytes.fromhex(salt)), bytes.fromhex(digest))
    except ValueError:
        return False


def create_account(database: PortalDatabase, username: str, role: str, password: str) -> None:
    if _USERNAME.fullmatch(username) is None:
        raise ValueError('Use letters, digits, dots, underscores or hyphens, starting with a letter or digit.')
    if role not in ROLES:
        raise ValueError('Choose a role defined by the local portal.')
    password_hash = hash_password(password)
    with database.session() as session:
        session.add(PortalUser(username=username, role_code=role, password_hash=password_hash))
        try:
            session.commit()
        except IntegrityError:
            raise ValueError('The account could not be created. Check whether the username is already in use.') from None


def update_account(database: PortalDatabase, username: str, *, password: str | None = None) -> None:
    """Reset a password or disable an account and revoke all of its sessions."""
    password_hash = hash_password(password) if password is not None else None
    with database.session() as session:
        user = session.scalar(select(PortalUser).where(PortalUser.username == username).with_for_update())
        if user is None:
            raise ValueError('The account does not exist.')
        if password_hash is None:
            user.is_active = False
        else:
            user.password_hash = password_hash
            user.failed_attempts = 0
            user.locked_until = None
        session.execute(delete(PortalSession).where(PortalSession.user_id == user.id))
        session.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('create', help='Create a portal account and prompt for its password')
    create.add_argument('username')
    create.add_argument('role', choices=ROLES)
    for name, help_text in (
        ('reset-password', 'Set a new password and revoke sessions'),
        ('disable', 'Disable an account and revoke sessions'),
    ):
        commands.add_parser(name, help=help_text).add_argument('username')
    args = parser.parse_args()

    database = None
    try:
        settings = DatabaseSettings.from_environment()
        if settings is None:
            raise ValueError('Configure the portal database before provisioning an account.')
        database = PortalDatabase(settings)
        if not database.is_ready():
            raise ValueError('The portal database is unavailable or requires migration.')
        if args.command == 'disable':
            update_account(database, args.username)
            print('Portal account disabled.')
            return
        if not sys.stdin.isatty():
            raise ValueError('Open an interactive terminal to enter the password privately.')
        with warnings.catch_warnings():
            warnings.simplefilter('error', GetPassWarning)
            password = getpass('New password (at least 12 characters): ')
            if password != getpass('Confirm password: '):
                raise ValueError('The passwords do not match.')
        if args.command == 'create':
            create_account(database, args.username, args.role, password)
        else:
            update_account(database, args.username, password=password)
        print('Portal account saved.')
    except (ValueError, EOFError) as error:
        parser.exit(1, str(error) or 'Password entry ended.\n')
    except GetPassWarning:
        parser.exit(1, 'The terminal could not disable password echo.\n')
    except SQLAlchemyError:
        parser.exit(1, 'The portal database operation failed.\n')
    finally:
        if database is not None:
            database.dispose()


if __name__ == '__main__':
    main()
