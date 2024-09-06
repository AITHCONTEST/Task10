
import { observer } from "mobx-react-lite";
import styles from "./styles/header.module.scss"


const Header = observer(() => {

  
    return (
      <header className={styles.header}>
        <div className={styles.main}>
          Translate2Win
        </div>

        <div className={styles.title}> Русско-Мансийский Переводчик </div>
      </header>
    );
  });
  
  export default Header;
  